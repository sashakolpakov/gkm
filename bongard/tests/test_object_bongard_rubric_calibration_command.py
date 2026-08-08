"""Production-boundary tests for the prose-rubric calibration command."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from threading import Lock

import pytest

from bongard.object_bongard_rubric_calibration import (
    run_object_bongard_rubric_calibration_observations,
)
from bongard.object_bongard_rubric_calibration_command import (
    ASSESSMENT_FILENAME,
    AUTHORIZATION_FILENAME,
    CALIBRATION_ACCEPTANCE_RULE,
    DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
    INVENTORY_FILENAME,
    PRECOMMIT_FILENAME,
    REPLAY_FILENAME,
    ObjectBongardRubricCalibrationAuthorization,
    ObjectBongardRubricCalibrationCommandError,
    load_object_bongard_rubric_calibration_authorization,
    load_object_bongard_rubric_calibration_execution_precommit,
    run_object_bongard_rubric_calibration_command,
)
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


def _fake_transport(source, calls: list[tuple[str, str]]):
    lock = Lock()
    by_png = {item.png_sha256: item for item in source.panels}
    first_rubric = source.rubric_specs[0].rubric

    def transport(prompt, paths, names, schema, **_kwargs):
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel = by_png[panel_digest]
        sheet = next(
            item
            for item in panel.hypothesis_packet.atlas_sheets
            if item.name == names[1]
        )
        if first_rubric in prompt:
            level = 4 if panel in source.group_a_panels else 0
        else:
            level = 4 if panel in source.group_b_panels else 0
        payload = {
            "scene": {"lower": level, "upper": level},
            "slots": [
                {"slot_id": slot.slot_id, "lower": level, "upper": level}
                for slot in sheet.slots
            ],
        }
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
            reasoning_effort="medium",
            names=names,
        )
        with lock:
            calls.append((panel.panel_id, sheet.name))
        return CodexStructuredResult(payload, receipt)

    return transport


def test_command_seals_before_calls_persists_everything_and_cold_replays(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "calibration"
    catalog, attestation = canonical_no_tools_runtime(
        DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256
    )
    calls: list[tuple[str, str]] = []
    causal_gate_seen = False

    def runner(source, **kwargs):
        nonlocal causal_gate_seen
        assert (output_root / AUTHORIZATION_FILENAME).is_file()
        assert (output_root / PRECOMMIT_FILENAME).is_file()
        authorization = load_object_bongard_rubric_calibration_authorization(
            output_root
        )
        precommit = load_object_bongard_rubric_calibration_execution_precommit(
            output_root
        )
        assert precommit.authorization_digest == authorization.authorization_digest
        assert len(authorization.jobs) == 24
        assert sum(len(item.sheets) for item in authorization.jobs) == 30
        causal_gate_seen = True
        return run_object_bongard_rubric_calibration_observations(
            source,
            **{
                **kwargs,
                "underlying_transport": _fake_transport(source, calls),
            },
        )

    result = run_object_bongard_rubric_calibration_command(
        output_root,
        cloud_policy_cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        model_catalog_snapshotter=lambda: catalog,
        launcher_fingerprinter=lambda _executable, **_kwargs: {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
        },
        runtime_attester=lambda **_kwargs: attestation,
        observation_runner=runner,
        underlying_transport=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsealed transport was called")
        ),
    )

    assert causal_gate_seen is True
    assert len(calls) == 30
    assert result.inventory.fresh_model_call_count == 30
    assert result.inventory.reused_model_call_count == 0
    assert result.replay.survivor_counts == (8, 8)
    assert result.accepted is True
    assert result.replay.to_data()["acceptance_rule"] == CALIBRATION_ACCEPTANCE_RULE
    assert result.replay.to_data()["threshold_tuning_performed"] is False
    assert result.replay.to_data()["preferred_candidate_selected"] is False
    assert result.replay.to_data()["fresh_broad_release_prepared"] is False
    assert len(tuple((output_root / "observer_artifacts").glob("*.json"))) == 24
    assert len(tuple((output_root / "journals").glob("**/manifest.json"))) == 30
    for filename in (
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        INVENTORY_FILENAME,
        ASSESSMENT_FILENAME,
        REPLAY_FILENAME,
    ):
        assert (output_root / filename).is_file()

    tampered = result.authorization.to_data()
    tampered["sheet_journal_count"] = 29
    with pytest.raises(
        ObjectBongardRubricCalibrationCommandError, match="policy"
    ):
        ObjectBongardRubricCalibrationAuthorization.from_data(tampered)


def test_command_source_has_no_lean_import() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "object_bongard_rubric_calibration_command.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any("lean" in name.lower() for name in imports)
