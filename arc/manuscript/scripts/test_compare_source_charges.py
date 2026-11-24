from __future__ import annotations

import json

import pytest

import compare_source_charges as comparison


def _state(level: int, lines: int, tier: str = comparison.MANIFEST_TIER) -> comparison.State:
    return comparison.State(
        level, tier,
        {"legs.py": b"pass\n" * lines, "players.py": b"from legs import *\n", "solve.py": b""},
        {},
    )


def _manifest_archive(*, reconstruction: bool = False) -> tuple[str, dict[str, bytes]]:
    path = "toy_legs/promotion_evidence/level_01/manifest.json"
    prefix = path.removesuffix("manifest.json") + "files/"
    # This would raise if imported. The audit may only parse/count these bytes.
    files = {
        "legs.py": b"raise AssertionError('solver source was executed')\nvalues = [1, 2, 3]\n",
        "players.py": b"from legs import *\n",
        "solve.py": b"pass\n",
        "checkpoint.json": json.dumps({
            "game": "toy", "reached": 1, "validated": True,
            "records": [{"level": 1, "marginal_C": 4, "reached": True}],
        }).encode(),
    }
    if reconstruction:
        files["provenance.json"] = json.dumps({
            "source_kind": "exact_path_reconstruction",
            "deterministic_reconstruction": True,
            "historical_source_boundary": False,
            "retained_historical_phase": False,
            "posthoc_acquisition_marginal_admissible": False,
        }).encode()
    manifest = {
        "schema": 2 if reconstruction else 1, "game": "toy", "level": 1,
        "validated": True, "taint_verdict": "clean",
        "promoted_files_sha256": {name: comparison.sha256(raw) for name, raw in files.items()},
    }
    archive = {prefix + name: raw for name, raw in files.items()}
    archive[path] = json.dumps(manifest).encode()
    return path, archive


def test_manifest_hashes_bind_retained_bytes_without_executing_them() -> None:
    path, archive = _manifest_archive()
    state, _ = comparison.load_manifest_state(archive.__getitem__, path, "toy", 1)
    assert state is not None
    assert state.tier == comparison.MANIFEST_TIER
    assert state.costs == {"legs.py": 5, "players.py": 1}
    archive[path.removesuffix("manifest.json") + "files/legs.py"] += b"pass\n"
    with pytest.raises(ValueError, match="hash mismatch"):
        comparison.load_manifest_state(archive.__getitem__, path, "toy", 1)


def test_certified_reconstruction_is_not_a_historical_retained_state() -> None:
    path, archive = _manifest_archive(reconstruction=True)
    state, proof = comparison.load_manifest_state(archive.__getitem__, path, "toy", 1)
    assert state is None
    assert proof["tier"] == "excluded_reconstruction_capsule"
    assert proof["certification_provenance"]["historical_source_boundary"] is False


def test_growth_excludes_g1_and_does_not_bridge_source_gaps() -> None:
    states = [_state(1, 10), None, _state(3, 30), _state(4, 40)]
    assert comparison.growth_series(states, primary=True) == [None, None, None, 10]
    with pytest.raises(ValueError, match="level positions"):
        comparison.growth_series([_state(1, 10), _state(3, 30)], primary=True)


def test_phase_candidates_are_separate_from_manifest_primary_scope() -> None:
    states = [_state(1, 10), _state(2, 20, comparison.FALLBACK_TIER), _state(3, 35)]
    assert comparison.growth_series(states, primary=True) == [None, None, None]
    assert comparison.growth_series(states, primary=False) == [None, 10, 15]


def test_paired_reversals_share_domains_without_bridging_missing_levels() -> None:
    pair = comparison.paired_metrics(
        [999, 5, 4, 999, 4, 5], [None, 5, 4, None, 4, 5]
    )
    assert pair["segments"] == [[2, 3], [5, 6]]
    assert pair["recorded"]["direction_reversals"] == 0
    assert pair["growth"]["direction_reversals"] == 0
    assert pair["recorded"]["strict_local_peaks"] == 0
    with pytest.raises(ValueError, match="level inventory"):
        comparison.paired_metrics([1, 2], [None])


def test_flats_are_ignored_inside_a_segment_but_are_not_strict_extrema() -> None:
    pair = comparison.paired_metrics([99, 5, 4, 4, 5], [None, 5, 4, 4, 5])
    assert pair["growth"]["direction_reversals"] == 1
    assert pair["growth"]["strict_local_troughs"] == 0


def test_phase_selection_excludes_pre_debrief_winning_source() -> None:
    root = "toy_legs"
    winning = root + "/wip_context/level_01/reached_before_debrief_win/metadata.json"
    retained = root + "/wip_context/level_01/after_debrief_retained/metadata.json"
    archive = {
        winning: json.dumps({"game": "toy", "level": 1, "phase": "reached_before_debrief", "reached": 1, "err": None}).encode(),
        retained: json.dumps({"game": "toy", "level": 1, "phase": "after_debrief", "reached": 1, "err": None, "created_at": "2000-01-01"}).encode(),
    }
    for name, raw in _state(1, 30).files.items():
        archive[retained.removesuffix("metadata.json") + "files/" + name] = raw
    state, _ = comparison.load_phase_candidate(archive.__getitem__, dict.fromkeys(archive, "oid"), root, "toy", 1)
    assert state is not None
    assert state.costs["legs.py"] == 30
    assert state.provenance["phase"] == "after_debrief"


def test_valid_parent_hash_cannot_skip_the_selected_adjacent_manifest(monkeypatch) -> None:
    original_read = comparison.FrozenBlobs.read
    root = comparison.HISTORY_PREFIX + "/ar25_legs"
    child_path = root + "/promotion_evidence/level_03/manifest.json"
    wrong_parent = "promotion_evidence/level_01/manifest.json"

    def read_with_wrong_parent(self, revision, path, oid=None):
        raw = original_read(self, revision, path, oid)
        if revision == comparison.HISTORY_REVISION and path == child_path:
            manifest = json.loads(raw)
            parent_raw = original_read(self, revision, root + "/" + wrong_parent)
            manifest["parent_manifest"] = wrong_parent
            manifest["parent_manifest_sha256"] = comparison.sha256(parent_raw)
            return json.dumps(manifest).encode()
        return raw

    monkeypatch.setattr(comparison.FrozenBlobs, "read", read_with_wrong_parent)
    with pytest.raises(ValueError, match="selected adjacent retained manifest"):
        comparison.build_report()


def test_frozen_gpt56_scope_and_examples() -> None:
    report = comparison.build_report()
    assert report["summary"]["physical_promotion_manifests"] == 164
    assert report["summary"]["historical_manifest_bound_states"] == 158
    assert report["summary"]["excluded_certification_capsules"] == 6
    assert report["summary"]["primary"]["paired_values"] == 134
    assert report["summary"]["primary"]["recorded_direction_reversals"] == 45
    assert report["summary"]["primary"]["growth_direction_reversals"] == 43
    rows = {row["game"]: row for row in report["games"]}
    assert rows["dc22"]["recorded_C"] == [49, 71, 96, 107, 2, 325]
    assert rows["dc22"]["primary_G"] == [None, 71, 96, 107, 202, 325]
    assert rows["dc22"]["paired"]["primary"]["recorded"]["direction_reversals"] == 2
    assert rows["dc22"]["paired"]["primary"]["growth"]["direction_reversals"] == 0
    assert rows["wa30"]["primary_G"] == [None, 20, 32, 50, 39, 23, 28, 34, 49]
    assert rows["ls20"]["primary_G"] == [None, 54, 86, 114, 138, 170, 158]
    assert rows["tr87"]["primary_G"] == [None] * 6
    assert rows["tr87"]["extended_G"] == [None] * 6
    assert rows["ft09"]["extended_G"][1:3] == [None, None]
    assert rows["sp80"]["extended_G"][3:5] == [None, None]
    assert all(row["primary_G"][0] is None and row["extended_G"][0] is None for row in rows.values())
    assert "\\label{tab:charge-growth-comparison}" in comparison.render_tex(report)
    assert len([line for line in comparison.render_markdown(report).splitlines() if line.startswith("| ")]) == 27
