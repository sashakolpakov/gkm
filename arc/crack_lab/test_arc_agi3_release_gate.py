from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

import arc_agi3_release_gate as R


INVENTORY = {
    "ar25": 8,
    "bp35": 9,
    "cd82": 6,
    "cn04": 6,
    "dc22": 6,
    "ft09": 6,
    "g50t": 7,
    "ka59": 7,
    "lf52": 10,
    "lp85": 8,
    "ls20": 7,
    "m0r0": 6,
    "r11l": 6,
    "re86": 8,
    "s5i5": 8,
    "sb26": 8,
    "sc25": 6,
    "sk48": 8,
    "sp80": 6,
    "su15": 9,
    "tn36": 7,
    "tr87": 6,
    "tu93": 9,
    "vc33": 7,
    "wa30": 9,
}
IDENTITY = {
    "campaign_id": "arc-agi3-contiguous-test",
    "release_name": "complete-183",
    "source_revision": "a" * 40,
    "created_at_utc": "2026-07-28T12:00:00Z",
}
CONTROL_BYTES = b"# frozen control contract\n"
CONTROL_SHA256 = hashlib.sha256(CONTROL_BYTES).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _checkpoint(game: str, level: int) -> dict:
    records = [
        {"level": current, "marginal_C": current, "reached": True}
        for current in range(1, level + 1)
    ]
    return {
        "game": game,
        "reached": level,
        "total_marginal_C": sum(
            record["marginal_C"] for record in records
        ),
        "records": records,
        "final_path": [1] * level,
        "validated": True,
    }


def _replay(
    *,
    kind: str,
    game: str,
    level: int,
    parent_checkpoint_sha256: str | None,
    checkpoint_sha256: str,
    winning_source_tree_sha256: str,
    exact_path_sha256: str,
    action_count: int,
) -> dict:
    return {
        "schema": 1,
        "kind": kind,
        "game": game,
        "target_level": level,
        "frontier_parent_level": level - 1,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "winning_source_tree_sha256": winning_source_tree_sha256,
        "exact_path_sha256": exact_path_sha256,
        "action_count": action_count,
        "observed_reached": level,
        "engine_sha256": CONTROL_SHA256,
        "result": "PASS",
    }


def _build_campaign(root: Path) -> tuple[Path, Path, dict[str, Path]]:
    environments = root / "environment_files"
    canonical = root / "canonical"
    environments.mkdir(parents=True)
    canonical.mkdir()

    for game, target in INVENTORY.items():
        version = environments / game / "version0001"
        version.mkdir(parents=True)
        _write_json(
            version / "metadata.json",
            {"baseline_actions": [0] * target, "game": game},
        )
        (version / f"{game}.py").write_text(
            "# public toolkit implementation is not release evidence\n",
            encoding="utf-8",
        )

        game_root = canonical / f"{game}_legs"
        evidence_root = game_root / "promotion_evidence"
        evidence_root.mkdir(parents=True)
        previous_checkpoint: str | None = None
        previous_manifest: str | None = None
        final_sources: dict[str, bytes] = {}
        final_checkpoint = b""

        for level in range(1, target + 1):
            boundary = evidence_root / f"level_{level:02d}"
            files = boundary / "files"
            transcripts = boundary / "transcripts"
            audits = boundary / "audits"
            files.mkdir(parents=True)
            transcripts.mkdir()
            audits.mkdir()

            checkpoint_value = _checkpoint(game, level)
            _write_json(files / "checkpoint.json", checkpoint_value)
            for source_name in sorted(R.REQUIRED_SOURCE_FILES):
                (files / source_name).write_text(
                    f"# exact {game} L{level} {source_name}\n",
                    encoding="utf-8",
                )
            (files / "acquisition.json").write_text(
                f'{{"game":"{game}","level":{level}}}',
                encoding="utf-8",
            )
            promoted = {
                path.name: _file_sha(path)
                for path in sorted(files.iterdir())
            }
            winning_source_files = sorted(
                name for name in promoted if name.endswith(".py")
            )
            winning_hashes = {
                name: promoted[name] for name in winning_source_files
            }
            winning_tree = _json_sha(winning_hashes)
            checkpoint_hash = promoted["checkpoint.json"]
            exact_path = checkpoint_value["final_path"]
            exact_path_hash = _json_sha(exact_path)

            transcript = transcripts / "host_turn.jsonl"
            transcript.write_text(
                json.dumps({"event": "turn", "game": game, "level": level})
                + "\n",
                encoding="utf-8",
            )
            transcript_hashes = {
                "transcripts/host_turn.jsonl": _file_sha(transcript)
            }
            primary = {
                **{
                    f"files/{name}": digest
                    for name, digest in promoted.items()
                },
                **transcript_hashes,
            }

            taint = {
                "schema": 1,
                "kind": "taint_audit",
                "game": game,
                "level": level,
                "scanner_sha256": CONTROL_SHA256,
                "checked_files_sha256": primary,
                "verdict": "PASS",
                "findings": [],
            }
            _write_json(audits / "taint.json", taint)
            _write_json(
                audits / "action_protocol.json",
                {
                    "schema": 1,
                    "kind": "action_protocol_audit",
                    "game": game,
                    "target_level": level,
                    "checkpoint_sha256": checkpoint_hash,
                    "exact_path_sha256": exact_path_hash,
                    "action_count": len(exact_path),
                    "runtime_enforcement":
                        "shared_violation_latch_across_root_and_clones",
                    "source_protocol_latch": "PASS",
                    "path_protocol_latch": "PASS",
                    "engine_sha256": CONTROL_SHA256,
                    "result": "PASS",
                },
            )
            for kind in ("path_replay", "source_replay"):
                _write_json(
                    audits / f"{kind}.json",
                    _replay(
                        kind=kind,
                        game=game,
                        level=level,
                        parent_checkpoint_sha256=previous_checkpoint,
                        checkpoint_sha256=checkpoint_hash,
                        winning_source_tree_sha256=winning_tree,
                        exact_path_sha256=exact_path_hash,
                        action_count=len(exact_path),
                    ),
                )
            hash_checked = {
                **primary,
                "audits/taint.json": _file_sha(audits / "taint.json"),
                "audits/action_protocol.json": _file_sha(
                    audits / "action_protocol.json"
                ),
                "audits/path_replay.json": _file_sha(
                    audits / "path_replay.json"
                ),
                "audits/source_replay.json": _file_sha(
                    audits / "source_replay.json"
                ),
            }
            _write_json(
                audits / "hash_audit.json",
                {
                    "schema": 1,
                    "kind": "hash_audit",
                    "game": game,
                    "level": level,
                    "hasher_sha256": CONTROL_SHA256,
                    "checked_files_sha256": hash_checked,
                    "result": "PASS",
                },
            )
            audit_bindings = {
                name: {
                    "path": relative,
                    "sha256": _file_sha(boundary / relative),
                }
                for name, relative in R.AUDIT_PATHS.items()
            }
            manifest = {
                "schema": R.BOUNDARY_MANIFEST_SCHEMA,
                "game": game,
                "level": level,
                "frontier": {
                    "parent_level": level - 1,
                    "target_level": level,
                    "parent_checkpoint_sha256": previous_checkpoint,
                },
                "parent_manifest": (
                    None
                    if level == 1
                    else {
                        "path": (
                            "promotion_evidence/"
                            f"level_{level - 1:02d}/manifest.json"
                        ),
                        "sha256": previous_manifest,
                    }
                ),
                "promoted_files_sha256": promoted,
                "winning_source_files": winning_source_files,
                "transcripts": [
                    {
                        "path": path,
                        "sha256": digest,
                    }
                    for path, digest in transcript_hashes.items()
                ],
                "audits": audit_bindings,
            }
            _write_json(boundary / "manifest.json", manifest)
            previous_checkpoint = checkpoint_hash
            previous_manifest = _file_sha(boundary / "manifest.json")
            final_checkpoint = (files / "checkpoint.json").read_bytes()
            final_sources = {
                name: (files / name).read_bytes()
                for name in winning_source_files
            }

        (game_root / "checkpoint.json").write_bytes(final_checkpoint)
        for source_name, content in final_sources.items():
            (game_root / source_name).write_bytes(content)
        (game_root / "README.md").write_text(
            f"Frozen canonical artifact for {game}.\n", encoding="utf-8"
        )

    control = root / "control.py"
    control.write_bytes(CONTROL_BYTES)
    return environments, canonical, {"control/control.py": control}


@pytest.fixture(scope="session")
def base_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-gate-base")
    return _build_campaign(root)


def _campaign_copy(
    base_campaign,
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    base_environments, base_canonical, base_controls = base_campaign
    environments = tmp_path / "environment_files"
    canonical = tmp_path / "canonical"
    shutil.copytree(base_environments, environments)
    shutil.copytree(base_canonical, canonical)
    original_control = next(iter(base_controls.values()))
    control = tmp_path / "control.py"
    shutil.copy2(original_control, control)
    return environments, canonical, {"control/control.py": control}


def _body(
    environments: Path,
    canonical: Path,
    controls: dict[str, Path],
) -> dict:
    return R.build_release_receipt_body(
        canonical_root=canonical,
        environments_root=environments,
        release_identity=IDENTITY,
        control_contract_files=controls,
    )


def _truncate_game(
    canonical: Path,
    *,
    game: str,
    claimed_level: int,
) -> None:
    game_root = canonical / f"{game}_legs"
    evidence = game_root / "promotion_evidence"
    for boundary in sorted(evidence.glob("level_*")):
        level = int(boundary.name.removeprefix("level_"))
        if level > claimed_level:
            shutil.rmtree(boundary)
    final = evidence / f"level_{claimed_level:02d}" / "files"
    (game_root / "checkpoint.json").write_bytes(
        (final / "checkpoint.json").read_bytes()
    )
    for source_name in sorted(R.REQUIRED_SOURCE_FILES):
        (game_root / source_name).write_bytes(
            (final / source_name).read_bytes()
        )


def _update_audit_binding(
    canonical: Path,
    game: str,
    level: int,
    audit_name: str,
) -> None:
    boundary = (
        canonical
        / f"{game}_legs"
        / "promotion_evidence"
        / f"level_{level:02d}"
    )
    manifest_path = boundary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = R.AUDIT_PATHS[audit_name]
    manifest["audits"][audit_name]["sha256"] = _file_sha(boundary / relative)
    _write_json(manifest_path, manifest)


def test_migration_diagnostic_passes_full_schema2_fixture_deterministically(
    base_campaign,
    tmp_path,
):
    environments, canonical, _ = _campaign_copy(base_campaign, tmp_path)
    first = R.diagnose_release_migration(
        canonical_root=canonical,
        environments_root=environments,
    )
    second = R.diagnose_release_migration(
        canonical_root=canonical,
        environments_root=environments,
    )

    assert first == second
    assert first["status"] == "PASS"
    assert first["summary"] == {
        "authoritative_games": 25,
        "authoritative_levels": 183,
        "canonical_reached_levels": 183,
        "schema2_candidate_boundaries": 183,
        "legacy_boundaries": 0,
        "missing_boundaries": 0,
        "invalid_boundaries": 0,
        "queued_boundaries": 0,
    }
    assert first["migration_queue"] == []
    assert len(first["games"]) == 25
    assert all(
        len(game["levels"]) == game["target_levels"]
        for game in first["games"].values()
    )


def test_migration_diagnostic_enumerates_legacy_missing_and_mutable_entries(
    base_campaign,
    tmp_path,
):
    environments, canonical, _ = _campaign_copy(base_campaign, tmp_path)
    (canonical / ".campaign_locks").mkdir()
    (canonical / "ar25_legs" / "wip_context").mkdir()
    missing = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_04"
    )
    shutil.rmtree(missing)
    legacy_path = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_03"
        / "manifest.json"
    )
    _write_json(
        legacy_path,
        {
            "schema": 1,
            "game": "ar25",
            "level": 3,
            "validated": True,
            "taint_verdict": "clean",
        },
    )

    first = R.diagnose_release_migration(
        canonical_root=canonical,
        environments_root=environments,
    )
    second = R.diagnose_release_migration(
        canonical_root=canonical,
        environments_root=environments,
    )
    assert first == second
    assert first["status"] == "FAIL"
    assert first["root_issues"] == [{
        "code": "mutable_non_evidence_root_entry",
        "path": ".campaign_locks",
        "detail": "entry is outside the frozen authoritative game trees",
    }]
    ar25 = first["games"]["ar25"]
    assert {
        issue["code"] for issue in ar25["issues"]
    } == {"mutable_non_evidence_entry"}
    assert ar25["levels"]["03"]["status"] == "legacy"
    assert {
        issue["code"] for issue in ar25["levels"]["03"]["issues"]
    } >= {"legacy_manifest_schema", "boolean_only_gate_claim"}
    assert ar25["levels"]["04"]["status"] == "missing"
    queue = {
        (row["game"], row["level"]): row
        for row in first["migration_queue"]
    }
    assert queue[("ar25", 3)]["status"] == "legacy"
    assert queue[("ar25", 4)]["status"] == "missing"
    assert queue[("ar25", 5)]["status"] == "invalid"
    assert "parent_frontier_discontinuity" in queue[
        ("ar25", 5)
    ]["issue_codes"]
    assert first["summary"]["legacy_boundaries"] == 1
    assert first["summary"]["missing_boundaries"] == 1
    assert first["summary"]["invalid_boundaries"] == 1
    assert first["summary"]["queued_boundaries"] == 3


def test_issues_and_reverifies_content_addressed_183_receipt(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    receipt = R.issue_release_receipt(
        canonical_root=canonical,
        environments_root=environments,
        receipt_directory=tmp_path / "receipts",
        release_identity=IDENTITY,
        control_contract_files=controls,
    )

    assert receipt.path.stem == hashlib.sha256(
        receipt.path.read_bytes()
    ).hexdigest()
    assert receipt.sha256 == receipt.path.stem
    assert receipt.body["canonical_game_count"] == 25
    assert receipt.body["authoritative_level_count"] == 183
    assert sum(len(rows) for rows in receipt.body["evidence"].values()) == 183
    assert receipt.path.stat().st_nlink == 1
    assert stat.S_IMODE(receipt.path.stat().st_mode) & 0o222 == 0
    assert R.verify_release_receipt(
        receipt_path=receipt.path,
        canonical_root=canonical,
        environments_root=environments,
        control_contract_files=controls,
    ).body == receipt.body


def test_partial_freeze_binds_181_claims_and_two_explicit_gaps(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    _truncate_game(canonical, game="lf52", claimed_level=8)

    body = R.build_partial_release_receipt_body(
        canonical_root=canonical,
        environments_root=environments,
        release_identity={
            **IDENTITY,
            "release_name": "strongest-partial-181",
        },
        expected_claimed_levels=181,
        control_contract_files=controls,
    )

    assert body["kind"] == "partial_campaign_freeze"
    assert body["complete"] is False
    assert body["authoritative_level_count"] == 183
    assert body["claimed_level_count"] == 181
    assert body["claimed_inventory"]["lf52"] == 8
    assert body["unclaimed_boundaries"] == [
        {"game": "lf52", "level": 9},
        {"game": "lf52", "level": 10},
    ]
    assert sum(len(rows) for rows in body["evidence"].values()) == 181
    with pytest.raises(R.ReleaseGateError):
        R.build_release_receipt_body(
            canonical_root=canonical,
            environments_root=environments,
            release_identity=IDENTITY,
            control_contract_files=controls,
        )

    receipt = R.issue_partial_release_receipt(
        canonical_root=canonical,
        environments_root=environments,
        receipt_directory=tmp_path / "partial-receipts",
        release_identity={
            **IDENTITY,
            "release_name": "strongest-partial-181",
        },
        expected_claimed_levels=181,
        control_contract_files=controls,
    )
    assert R.verify_partial_release_receipt(
        receipt_path=receipt.path,
        canonical_root=canonical,
        environments_root=environments,
        control_contract_files=controls,
    ).body == receipt.body


def test_partial_freeze_fails_closed_on_wrong_count_or_internal_hole(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    _truncate_game(canonical, game="lf52", claimed_level=8)
    with pytest.raises(
        R.ReleaseGateError,
        match="frontier count mismatch",
    ):
        R.build_partial_release_receipt_body(
            canonical_root=canonical,
            environments_root=environments,
            release_identity=IDENTITY,
            expected_claimed_levels=180,
            control_contract_files=controls,
        )

    shutil.rmtree(
        canonical
        / "lf52_legs"
        / "promotion_evidence"
        / "level_07"
    )
    with pytest.raises(R.ReleaseGateError, match="evidence levels"):
        R.build_partial_release_receipt_body(
            canonical_root=canonical,
            environments_root=environments,
            release_identity=IDENTITY,
            expected_claimed_levels=181,
            control_contract_files=controls,
        )


def test_action_protocol_audit_requires_shared_clone_latch() -> None:
    value = {
        "schema": 1,
        "kind": "action_protocol_audit",
        "game": "ar25",
        "target_level": 1,
        "checkpoint_sha256": "a" * 64,
        "exact_path_sha256": "b" * 64,
        "action_count": 1,
        "runtime_enforcement":
            "shared_violation_latch_across_root_and_clones",
        "source_protocol_latch": "PASS",
        "path_protocol_latch": "PASS",
        "engine_sha256": CONTROL_SHA256,
        "result": "PASS",
    }
    R._validate_action_protocol_audit(
        value,
        game="ar25",
        level=1,
        checkpoint_sha256="a" * 64,
        exact_path_sha256="b" * 64,
        action_count=1,
        allowed_tool_hashes=frozenset({CONTROL_SHA256}),
    )
    value["source_protocol_latch"] = "FAIL"
    with pytest.raises(R.ReleaseGateError):
        R._validate_action_protocol_audit(
            value,
            game="ar25",
            level=1,
            checkpoint_sha256="a" * 64,
            exact_path_sha256="b" * 64,
            action_count=1,
            allowed_tool_hashes=frozenset({CONTROL_SHA256}),
        )


def test_reverification_fails_after_each_evidence_mutation_or_deletion(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    receipt = R.issue_release_receipt(
        canonical_root=canonical,
        environments_root=environments,
        receipt_directory=tmp_path / "receipts",
        release_identity=IDENTITY,
        control_contract_files=controls,
    )
    boundary = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_01"
    )
    targets = [
        boundary / "files" / "legs.py",
        boundary / "files" / "checkpoint.json",
        boundary / "transcripts" / "host_turn.jsonl",
        boundary / "audits" / "taint.json",
        boundary / "audits" / "action_protocol.json",
        boundary / "audits" / "path_replay.json",
        boundary / "audits" / "hash_audit.json",
        boundary / "manifest.json",
        canonical / "ar25_legs" / "README.md",
    ]
    for target in targets:
        original = target.read_bytes()
        original_mode = stat.S_IMODE(target.stat().st_mode)
        target.write_bytes(original + b"x")
        with pytest.raises(R.ReleaseGateError):
            R.verify_release_receipt(
                receipt_path=receipt.path,
                canonical_root=canonical,
                environments_root=environments,
                control_contract_files=controls,
            )
        target.write_bytes(original)
        target.chmod(original_mode)
        R.verify_release_receipt(
            receipt_path=receipt.path,
            canonical_root=canonical,
            environments_root=environments,
            control_contract_files=controls,
        )

        target.unlink()
        with pytest.raises(R.ReleaseGateError):
            R.verify_release_receipt(
                receipt_path=receipt.path,
                canonical_root=canonical,
                environments_root=environments,
                control_contract_files=controls,
            )
        target.write_bytes(original)
        target.chmod(original_mode)
        R.verify_release_receipt(
            receipt_path=receipt.path,
            canonical_root=canonical,
            environments_root=environments,
            control_contract_files=controls,
        )


def test_reverification_binds_inventory_control_and_receipt_inode(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    receipt = R.issue_release_receipt(
        canonical_root=canonical,
        environments_root=environments,
        receipt_directory=tmp_path / "receipts",
        release_identity=IDENTITY,
        control_contract_files=controls,
    )

    control = controls["control/control.py"]
    original_control = control.read_bytes()
    control.write_bytes(original_control + b"# changed\n")
    with pytest.raises(R.ReleaseGateError):
        R.verify_release_receipt(
            receipt_path=receipt.path,
            canonical_root=canonical,
            environments_root=environments,
            control_contract_files=controls,
        )
    control.write_bytes(original_control)

    metadata = environments / "ar25" / "version0001" / "metadata.json"
    original_metadata = metadata.read_bytes()
    metadata.write_bytes(original_metadata + b" ")
    with pytest.raises(R.ReleaseGateError):
        R.verify_release_receipt(
            receipt_path=receipt.path,
            canonical_root=canonical,
            environments_root=environments,
            control_contract_files=controls,
        )
    metadata.write_bytes(original_metadata)

    alias = tmp_path / "receipt-alias.json"
    os.link(receipt.path, alias)
    with pytest.raises(R.ReleaseGateError, match="hard-linked"):
        R.verify_release_receipt(
            receipt_path=receipt.path,
            canonical_root=canonical,
            environments_root=environments,
            control_contract_files=controls,
        )


def test_rejects_missing_intermediate_and_extra_level_boundaries(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    missing = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_04"
    )
    shutil.rmtree(missing)
    with pytest.raises(R.ReleaseGateError, match="not exactly"):
        _body(environments, canonical, controls)

    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path / "extra"
    )
    source = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_08"
    )
    shutil.copytree(source, source.parent / "level_09")
    with pytest.raises(R.ReleaseGateError, match="not exactly"):
        _body(environments, canonical, controls)


def test_rejects_multi_level_overshoot_even_when_audit_hash_is_updated(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    replay_path = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_03"
        / R.AUDIT_PATHS["path_replay"]
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["observed_reached"] = 4
    _write_json(replay_path, replay)
    _update_audit_binding(canonical, "ar25", 3, "path_replay")

    with pytest.raises(R.ReleaseGateError, match="inexact"):
        _body(environments, canonical, controls)


def test_rejects_boolean_only_gate_claim_even_when_manifest_hash_matches(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    taint_path = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_03"
        / R.AUDIT_PATHS["taint"]
    )
    _write_json(taint_path, {"passed": True})
    _update_audit_binding(canonical, "ar25", 3, "taint")

    with pytest.raises(R.ReleaseGateError, match="schema mismatch"):
        _body(environments, canonical, controls)


def test_rejects_stale_parent_frontier_even_with_rehashed_manifest(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    manifest_path = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_03"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frontier"]["parent_checkpoint_sha256"] = "f" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(R.ReleaseGateError, match="parent/frontier"):
        _body(environments, canonical, controls)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_rejects_links_and_nonregular_entries(
    base_campaign,
    tmp_path,
    kind,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    transcript = (
        canonical
        / "ar25_legs"
        / "promotion_evidence"
        / "level_01"
        / "transcripts"
        / "host_turn.jsonl"
    )
    if kind == "symlink":
        target = tmp_path / "outside.log"
        target.write_text("outside\n", encoding="utf-8")
        transcript.unlink()
        transcript.symlink_to(target)
    elif kind == "hardlink":
        os.link(transcript, transcript.with_name("alias.jsonl"))
    else:
        transcript.unlink()
        os.mkfifo(transcript)

    with pytest.raises(R.ReleaseGateError):
        _body(environments, canonical, controls)


def test_rejects_missing_or_extra_canonical_games_and_mutable_lock_tree(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    shutil.rmtree(canonical / "ar25_legs")
    with pytest.raises(R.ReleaseGateError, match="authoritative games"):
        _body(environments, canonical, controls)

    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path / "extra"
    )
    (canonical / ".campaign_locks").mkdir()
    with pytest.raises(R.ReleaseGateError, match="extra"):
        _body(environments, canonical, controls)


def test_rejects_inventory_with_extra_game_or_wrong_level_total(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    extra = environments / "zz99" / "version0001"
    extra.mkdir(parents=True)
    _write_json(extra / "metadata.json", {"baseline_actions": [0]})
    with pytest.raises(R.ReleaseGateError, match="exactly 25 games"):
        _body(environments, canonical, controls)

    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path / "total"
    )
    metadata = environments / "ar25" / "version0001" / "metadata.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["baseline_actions"].append(0)
    _write_json(metadata, value)
    with pytest.raises(R.ReleaseGateError, match="total 183"):
        _body(environments, canonical, controls)


def test_rejects_stale_top_level_checkpoint_and_source(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    top_source = canonical / "ar25_legs" / "legs.py"
    top_source.write_text("# stale source\n", encoding="utf-8")
    with pytest.raises(R.ReleaseGateError, match="canonical source is stale"):
        _body(environments, canonical, controls)

    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path / "checkpoint"
    )
    top_checkpoint = canonical / "ar25_legs" / "checkpoint.json"
    value = json.loads(top_checkpoint.read_text(encoding="utf-8"))
    value["final_path"].append(1)
    _write_json(top_checkpoint, value)
    with pytest.raises(R.ReleaseGateError, match="canonical checkpoint"):
        _body(environments, canonical, controls)


def test_receipt_store_is_exclusive_and_outside_canonical_tree(
    base_campaign,
    tmp_path,
):
    environments, canonical, controls = _campaign_copy(
        base_campaign, tmp_path
    )
    kwargs = {
        "canonical_root": canonical,
        "environments_root": environments,
        "release_identity": IDENTITY,
        "control_contract_files": controls,
    }
    receipt = R.issue_release_receipt(
        receipt_directory=tmp_path / "receipts", **kwargs
    )
    with pytest.raises(R.ReleaseGateError, match="already exists"):
        R.issue_release_receipt(
            receipt_directory=tmp_path / "receipts", **kwargs
        )
    assert receipt.path.exists()

    with pytest.raises(R.ReleaseGateError, match="outside"):
        R.issue_release_receipt(
            receipt_directory=canonical / "receipts", **kwargs
        )
