from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from bongard.historical_exposure import (
    ABSTRACT_PAIR_PARTITION_NAMESPACE,
    BASIC_PARTITION_NAMESPACE,
    HistoricalExposureError,
    _RepositoryEvidenceReader,
    build_historical_exposure,
    load_historical_exposure,
    verify_historical_exposure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = REPO_ROOT / "bongard/data/historical_exposure_v1.json"


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pair_line_digest(values: list[list[str]]) -> str:
    payload = "".join(f"{first}\t{second}\n" for first, second in values).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_persisted_seed_is_exact_reconstruction_of_audited_artifacts():
    persisted = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    assert build_historical_exposure(REPO_ROOT) == persisted

    seed = load_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)
    assert seed.seed_digest == persisted["seed_digest"]
    assert len(seed.basic_shape_families) == 178
    assert len(seed.visual_basic_shape_families) == 66
    assert len(set(seed.basic_shape_families) & set(seed.visual_basic_shape_families)) == 66

    audited = verify_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)
    assert audited.seed_digest == seed.seed_digest


def test_normal_load_is_self_contained_after_legacy_cleanup(tmp_path: Path):
    seed = load_historical_exposure(
        SEED_PATH,
        repo_root=tmp_path / "deleted-repository",
        dataset_root=tmp_path / "deleted-generator",
    )
    assert len(seed.partition.drill) == 300
    assert len(seed.abstract_partition.drill) == 85
    assert len(seed.admissible_abstract_pairs) == 194


def test_basic_partition_is_family_clean_and_has_frozen_witnesses():
    seed = load_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)
    partition = seed.partition

    assert partition.namespace == BASIC_PARTITION_NAMESPACE
    assert partition.counts == {"drill": 300, "dev": 75, "sealed": 74}
    assert len(seed.unused_basic_shape_families) == 449
    assert not set(seed.basic_shape_families) & set(seed.unused_basic_shape_families)
    assert len(set(partition.eligible)) == 449
    assert not set(partition.drill) & set(partition.dev)
    assert not set(partition.drill) & set(partition.sealed)
    assert not set(partition.dev) & set(partition.sealed)

    assert partition.drill[:10] == (
        "trapez_parallelogram",
        "thin_parallel_bridge",
        "thin_seven_lines4",
        "inverse_trapez_parallel",
        "thin_symm_band",
        "inverse_symm_sharp_axe",
        "open_band_three_arcs2",
        "inverse_sector_arc",
        "parallel_sector3",
        "open_uneven_band_four_arcs3",
    )
    assert partition.dev[:3] == (
        "thin_seven_lines3",
        "two_mismatch_sectors3",
        "inverse_trap_arc180",
    )
    assert partition.sealed[:3] == (
        "symm_trans_arc_lamp",
        "open_line_s4",
        "irregular_jar_dagger3",
    )
    assert partition.drill_digest == (
        "sha256:c4944841895c61cc0337e846888a1f7154798351d6d07ac6cf9512795d2ec173"
    )
    assert partition.dev_digest == (
        "sha256:b5c57203bb9851ad596dd94036b4c26ff2776556775577b7637e45b8f901741c"
    )
    assert partition.sealed_digest == (
        "sha256:a912268d1a9da8cd3254abfefd7b7d0b8fab027c2f06aa800e1bab4abff97d0f"
    )


def test_abstract_semantics_and_freeform_uncertainty_remain_explicit():
    seed = load_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)

    assert len(seed.abstract_attributes) == 26
    assert len(seed.visual_abstract_attributes) == 16
    assert len(seed.abstract_pairs) == 67
    assert len(seed.admissible_abstract_pairs) == 194
    assert len(seed.unused_abstract_pairs) == 127
    assert seed.abstract_partition.namespace == ABSTRACT_PAIR_PARTITION_NAMESPACE
    assert seed.abstract_partition.counts == {"drill": 85, "dev": 21, "sealed": 21}
    assert len(set(seed.abstract_partition.eligible)) == 127
    assert not set(seed.abstract_pairs) & set(seed.abstract_partition.eligible)
    assert set(seed.abstract_pairs) | set(seed.abstract_partition.eligible) == set(
        seed.admissible_abstract_pairs
    )
    assert seed.abstract_partition.drill[:3] == (
        ("has_seven_straight_lines", "exist_triangle"),
        ("exist_regular", "exist_quadrangle"),
        ("has_six_straight_lines", "exist_triangle"),
    )
    assert seed.abstract_partition.dev[:3] == (
        ("has_line_crossing", "exist_regular"),
        ("has_two_parts", "exist_regular"),
        ("has_seven_straight_lines", "has_line_crossing"),
    )
    assert seed.abstract_partition.sealed[:3] == (
        ("has_line_crossing", "has_two_parts"),
        ("has_five_straight_lines", "exist_triangle"),
        ("self_transposed", "has_seven_straight_lines"),
    )
    assert seed.abstract_partition.drill_digest == (
        "sha256:91c421e12453554ab23cbd5c4b8252cc1fa644f3c99b1411c4e5f7a23b2b910d"
    )
    assert seed.abstract_partition.dev_digest == (
        "sha256:1e61ea3e973ecbef8b9b29639710f148ebf5e70f87b6cda727deba0b679a6dd2"
    )
    assert seed.abstract_partition.sealed_digest == (
        "sha256:cec9217f1b9a4799473f8d8ac61c0eb50ef61701abdd0dbcc6c6e46c759f7e12"
    )
    assert set(seed.abstract_pairs) <= set(seed.admissible_abstract_pairs)
    assert seed.freeform_status == "indeterminate"
    assert seed.freeform_exact_task_ids == ("ff_nact6_0292",)


def test_exact_official_identity_exposure_is_not_inflated_from_generated_ids():
    seed = load_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)

    assert seed.exact_official_task_ids == (
        "ff_nact6_0292",
        "bd_isosceles_trapezoid-no_obtuse_angle_six_lines2_0000",
        "hd_convex_0004",
    )
    assert seed.exact_official_panel_ids == ()
    assert "bd_open_s5_0279" not in seed.exact_official_task_ids

    # Downloading or drilling these official training tasks happened after the
    # historical cutoff. Mere file presence must not rewrite prior exposure.
    assert not {
        "bd_trapez_parallelogram_0000",
        "ff_nact2_5_0000",
        "hd_balanced_two-exist_quadrangle_0000",
    } & set(seed.exact_official_task_ids)


def test_seed_digest_tampering_fails_closed(tmp_path: Path):
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    raw["seed"]["semantic_exposure"]["basic"]["shape_families"][0] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(HistoricalExposureError, match="digest mismatch"):
        load_historical_exposure(
            path,
            repo_root=REPO_ROOT,
            verify_evidence=False,
        )


def test_rehashed_abstract_pair_sibling_split_tampering_fails_closed(tmp_path: Path):
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    members = raw["seed"]["abstract_pair_partition"]["members"]
    members["dev"][0] = members["drill"][0]
    raw["seed_digest"] = _canonical_digest(raw["seed"])
    path = tmp_path / "cross-partition-pair.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(HistoricalExposureError, match="overlap"):
        load_historical_exposure(path, verify_evidence=False)


def test_rehashed_disjoint_abstract_pair_reshuffle_fails_deterministic_rank(
    tmp_path: Path,
) -> None:
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    partition = raw["seed"]["abstract_pair_partition"]
    members = partition["members"]
    members["drill"][0], members["dev"][0] = members["dev"][0], members["drill"][0]
    for cohort in ("drill", "dev", "sealed"):
        partition["digests"][cohort] = _pair_line_digest(members[cohort])
        partition["first_ids"][cohort] = members[cohort][:10]
    raw["seed_digest"] = _canonical_digest(raw["seed"])
    path = tmp_path / "reshuffled-pairs.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(HistoricalExposureError, match="deterministic hash ranking"):
        load_historical_exposure(path, verify_evidence=False)


def test_recomputed_envelope_cannot_override_repository_evidence(tmp_path: Path):
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    raw["seed"]["qualification"]["official_panel_bytes_evidenced"] = True
    raw["seed_digest"] = _canonical_digest(raw["seed"])
    path = tmp_path / "false-claim.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(HistoricalExposureError, match="differs from the audited"):
        load_historical_exposure(path, repo_root=REPO_ROOT, verify_evidence=True)


def test_audit_never_uses_png_as_an_evidence_source():
    seed = load_historical_exposure(SEED_PATH, repo_root=REPO_ROOT)
    assert all(not path.lower().endswith(".png") for path, _digest in seed.evidence_files)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _temporary_annotated_snapshot(tmp_path: Path) -> tuple[Path, str, str, str, bytes]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    relative_path = "bongard/legacy/evidence.json"
    evidence = repo / relative_path
    evidence.parent.mkdir(parents=True)
    payload = b'{"concept":"bird-like"}\n'
    evidence.write_bytes(payload)
    _git(repo, "add", relative_path)
    _git(
        repo,
        "-c",
        "user.name=Historical Audit",
        "-c",
        "user.email=audit@example.invalid",
        "commit",
        "-q",
        "-m",
        "legacy snapshot",
    )
    tag = "test-historical-snapshot"
    _git(
        repo,
        "-c",
        "user.name=Historical Audit",
        "-c",
        "user.email=audit@example.invalid",
        "tag",
        "-a",
        tag,
        "-m",
        "frozen historical evidence",
    )
    tag_object = _git(repo, "rev-parse", f"refs/tags/{tag}")
    commit = _git(repo, "rev-parse", f"{tag}^{{commit}}")
    evidence.unlink()
    return repo, tag, tag_object, commit, payload


def test_missing_legacy_file_is_read_from_exact_annotated_snapshot(tmp_path: Path):
    repo, tag, tag_object, commit, payload = _temporary_annotated_snapshot(tmp_path)
    relative_path = "bongard/legacy/evidence.json"
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    reader = _RepositoryEvidenceReader(
        repo,
        expected_digests={relative_path: expected},
        snapshot_tag=tag,
        snapshot_tag_object=tag_object,
        snapshot_commit=commit,
        snapshot_paths={relative_path},
        special_commits={},
    )

    assert reader.read_bytes(relative_path) == payload
    assert reader.address(relative_path) == expected


def test_pinned_fallback_rejects_hash_or_tag_tampering(tmp_path: Path):
    repo, tag, tag_object, commit, payload = _temporary_annotated_snapshot(tmp_path)
    relative_path = "bongard/legacy/evidence.json"
    wrong_digest = "sha256:" + "0" * 64
    wrong_hash_reader = _RepositoryEvidenceReader(
        repo,
        expected_digests={relative_path: wrong_digest},
        snapshot_tag=tag,
        snapshot_tag_object=tag_object,
        snapshot_commit=commit,
        snapshot_paths={relative_path},
        special_commits={},
    )
    with pytest.raises(HistoricalExposureError, match="evidence hash mismatch"):
        wrong_hash_reader.read_bytes(relative_path)

    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    wrong_tag_reader = _RepositoryEvidenceReader(
        repo,
        expected_digests={relative_path: expected},
        snapshot_tag=tag,
        snapshot_tag_object="0" * 40,
        snapshot_commit=commit,
        snapshot_paths={relative_path},
        special_commits={},
    )
    with pytest.raises(HistoricalExposureError, match="tag object"):
        wrong_tag_reader.read_bytes(relative_path)


def test_configured_legacy_fallback_objects_match_frozen_seed_hashes():
    seed = load_historical_exposure(SEED_PATH)
    expected = dict(seed.evidence_files)
    reader = _RepositoryEvidenceReader(REPO_ROOT, expected_digests=expected)
    fallback_paths = (
        "bongard/bongard_logo_report.md",
        "bongard/crack_lab/agent_solutions/logo_full_predicates/results.json",
        "bongard/crack_lab/semantic_grounded_runs/codex_eod_20260805_v1/campaign.json",
        "bongard/crack_lab/semantic_grounded_runs/codex_blind_bird6_20260905_v1/campaign.json",
        "bongard/crack_lab/semantic_hybrid_runs/codex_bird6_latent_20260905_v1/campaign.json",
        "bongard/run_bongard_logo_adapter.py",
    )
    for relative_path in fallback_paths:
        payload = reader._fallback_bytes(relative_path)
        actual = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert actual == expected[relative_path]


def test_explicit_audit_rebuilds_with_all_legacy_files_forced_to_pinned_git():
    frozen = load_historical_exposure(SEED_PATH)
    reader = _RepositoryEvidenceReader(
        REPO_ROOT,
        expected_digests=dict(frozen.evidence_files),
        prefer_pinned_legacy=True,
    )
    audited = verify_historical_exposure(
        SEED_PATH,
        repo_root=REPO_ROOT,
        _evidence_reader=reader,
    )
    assert audited.seed_digest == frozen.seed_digest
