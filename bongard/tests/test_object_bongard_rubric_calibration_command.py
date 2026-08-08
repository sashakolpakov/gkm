"""Production-boundary tests for the prose-rubric calibration command."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from threading import Lock

import pytest

from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    run_object_bongard_rubric_calibration_observations,
)
from bongard.object_bongard_rubric_calibration_command import (
    ASSESSMENT_FILENAME,
    AUTHORIZATION_FILENAME,
    CALIBRATION_ACCEPTANCE_RULE,
    CALIBRATION_JOB_COUNT,
    CALIBRATION_SHEET_JOURNAL_COUNT,
    DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
    INVENTORY_FILENAME,
    NOMINATION_DIRECTORY,
    PRECOMMIT_FILENAME,
    REPLAY_FILENAME,
    CalibrationObservationJobCommitment,
    ObjectBongardRubricCalibrationAuthorization,
    ObjectBongardRubricCalibrationCommandError,
    load_object_bongard_rubric_calibration_authorization,
    load_object_bongard_rubric_calibration_execution_precommit,
    run_object_bongard_rubric_calibration_command,
    verify_object_bongard_rubric_calibration_command_directory,
)
from bongard.object_bongard_rubric_nomination_command import (
    DEFAULT_EXPECTED_LAUNCHER_SHA256,
    run_object_bongard_rubric_nomination,
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

    def transport(prompt, paths, names, schema, **_kwargs):
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel = by_png[panel_digest]
        sheet = next(
            item
            for item in panel.hypothesis_packet.atlas_sheets
            if item.name == names[1]
        )
        level = 4 if panel in source.group_a_panels else 0
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


def test_command_cardinality_is_one_signed_spec() -> None:
    assert CALIBRATION_JOB_COUNT == 12
    assert CALIBRATION_SHEET_JOURNAL_COUNT == 15
    assert "single-frozen-signed-rubric-spec" in CALIBRATION_ACCEPTANCE_RULE


def test_command_seals_before_calls_persists_everything_and_cold_replays(
    tmp_path: Path,
) -> None:
    nomination_root = tmp_path / "nomination"
    output_root = tmp_path / "calibration"
    catalog, attestation = canonical_no_tools_runtime(
        DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256
    )
    calls: list[tuple[str, str]] = []
    nomination_calls = 0
    causal_gate_seen = False

    def nomination_transport(prompt, paths, names, schema, **_kwargs):
        nonlocal nomination_calls
        nomination_calls += 1
        payload = {
            "profiles": [
                {
                    "group_id": "group_0",
                    "rubric": "Mismatched joined sector-like pieces recur.",
                    "feature_ids": ["paired_sector_mismatch_support_ppm"],
                },
                {
                    "group_id": "group_1",
                    "rubric": "A triangle accompanied by three spans recurs.",
                    "feature_ids": ["triangle_with_three_lines_support_ppm"],
                },
            ]
        }
        return CodexStructuredResult(
            payload,
            canonical_codex_receipt(
                prompt,
                paths,
                schema,
                payload,
                launcher_digest=DEFAULT_EXPECTED_LAUNCHER_SHA256,
                reasoning_effort="medium",
                names=names,
            ),
        )

    nomination = run_object_bongard_rubric_nomination(
        nomination_root,
        source_root=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
        cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        catalog_snapshotter=lambda: catalog,
        launcher_fingerprinter=lambda _executable, **_kwargs: {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": DEFAULT_EXPECTED_LAUNCHER_SHA256,
        },
        runtime_attester=lambda **_kwargs: attestation,
        visual_transport=nomination_transport,
    )
    assert nomination_calls == 1
    assert nomination.accepted is True

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
        assert len(authorization.jobs) == CALIBRATION_JOB_COUNT == 12
        assert (
            sum(len(item.sheets) for item in authorization.jobs)
            == CALIBRATION_SHEET_JOURNAL_COUNT
            == 15
        )
        assert tuple(item.rubric_spec_index for item in authorization.jobs) == (
            0,
        ) * CALIBRATION_JOB_COUNT
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
        nomination_root=nomination_root,
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
    assert (output_root / NOMINATION_DIRECTORY).is_dir()
    assert result.authorization.nomination_binding.to_data() == {
        "artifact_digest": nomination.artifact.artifact_digest,
        "authorization_digest": nomination.authorization_digest,
        "execution_precommit_digest": nomination.execution_precommit_digest,
        "cold_replay_digest": nomination.cold_replay_digest,
        "command_result_digest": nomination.result_digest,
    }
    assert len(calls) == 15
    assert result.inventory.fresh_model_call_count == 15
    assert result.inventory.reused_model_call_count == 0
    assert result.replay.survivor_counts == (2,)
    assert result.accepted is True
    assert result.replay.to_data()["acceptance_rule"] == CALIBRATION_ACCEPTANCE_RULE
    assert result.replay.to_data()["threshold_tuning_performed"] is False
    assert result.replay.to_data()["preferred_candidate_selected"] is False
    assert result.replay.to_data()["fresh_broad_release_prepared"] is False
    assert len(tuple((output_root / "observer_artifacts").glob("*.json"))) == 12
    assert len(tuple((output_root / "journals").glob("**/manifest.json"))) == 15
    for filename in (
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        INVENTORY_FILENAME,
        ASSESSMENT_FILENAME,
        REPLAY_FILENAME,
    ):
        assert (output_root / filename).is_file()

    assert (
        verify_object_bongard_rubric_calibration_command_directory(output_root)
        == result
    )

    tampered = result.authorization.to_data()
    tampered["sheet_journal_count"] = 14
    with pytest.raises(
        ObjectBongardRubricCalibrationCommandError, match="policy"
    ):
        ObjectBongardRubricCalibrationAuthorization.from_data(tampered)

    reverse_job = result.authorization.jobs[0].to_data()
    reverse_job["rubric_spec_index"] = 1
    with pytest.raises(
        ObjectBongardRubricCalibrationCommandError,
        match="canonical index zero",
    ):
        CalibrationObservationJobCommitment.from_data(reverse_job)


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
