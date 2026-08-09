"""Synthetic exactly-once test for the action-count phase command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_action_count_phase_command import (
    PLAN_SCHEMA,
    _truth_records,
    action_count_batch_output_schema,
    action_count_batch_prompt,
    run_action_count_phase,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASKS = ("hd_action_fixture_a_0000", "hd_action_fixture_b_0000")
STYLES = ("circle", "normal", "square", "triangle", "zigzag")


def _actions(straight: int, arcs: int, offset: int) -> list[str]:
    result = [
        f"line_{STYLES[(offset + index) % len(STYLES)]}_0.100-0.200"
        for index in range(straight)
    ]
    result.extend(
        f"arc_{STYLES[(offset + straight + index) % len(STYLES)]}_0.100-0.200"
        for index in range(arcs)
    )
    return result


def _fixture(tmp_path: Path):
    dataset = tmp_path / "ShapeBongard_V2"
    programs: dict[str, object] = {}
    truth_by_png: dict[str, tuple[int, int]] = {}
    seed = 0
    for task_offset, task_id in enumerate(TASKS):
        sides: list[object] = []
        for side_index, folder in enumerate((1, 0)):
            side: list[object] = []
            for panel_index in range(7):
                straight = (panel_index + side_index + task_offset) % 10
                arcs = min(9 - straight, (panel_index + task_offset) % 3)
                side.append([_actions(straight, arcs, seed)])
                payload = _png(seed)
                seed += 1
                path = (
                    dataset
                    / "hd/images"
                    / task_id
                    / str(folder)
                    / f"{panel_index}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                truth_by_png[hashlib.sha256(payload).hexdigest()] = (straight, arcs)
            sides.append(side)
        programs[task_id] = sides
    action_file = dataset / "hd/hd_action_programs.json"
    action_file.parent.mkdir(parents=True, exist_ok=True)
    action_raw = canonical_json(programs) + b"\n"
    action_file.write_bytes(action_raw)
    truth = _truth_records(programs, TASKS)
    plan = {
        "schema": PLAN_SCHEMA,
        "current_state": {
            "selected_panel_pixels_read": False,
            "model_calls_made": 0,
        },
        "calibration_authority": {
            "python_is_canonical_authority": True,
            "lean_required": False,
        },
        "dataset_bindings": {
            "hd_action_program_raw_sha256": (
                "sha256:" + hashlib.sha256(action_raw).hexdigest()
            ),
        },
        "cohorts": {
            "fit": {
                "task_ids": list(TASKS),
                "action_label_manifest_digest": "sha256:" + canonical_digest(truth),
            }
        },
    }
    plan["record_digest"] = "sha256:" + canonical_digest(plan)
    plan_file = tmp_path / "plan.json"
    plan_file.write_bytes(canonical_json(plan) + b"\n")
    return dataset, action_file, plan_file, plan, truth_by_png


def _runtime() -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        verbose=False,
        executable="/private/synthetic-codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest="2" * 64,
    )


def test_phase_batches_fourteen_neutral_views_before_labels_and_cold_replays(
    tmp_path: Path,
) -> None:
    dataset, action_file, plan_file, plan, truth_by_png = _fixture(tmp_path)
    calls: list[tuple[str, ...]] = []

    def transport(prompt, paths, names, schema, **_kwargs):
        assert tuple(names) == tuple(f"view_{index:02d}.png" for index in range(14))
        assert len(paths) == 14
        assert not any(task_id in prompt for task_id in TASKS)
        assert "closure or convexity judgment" in prompt
        payload: dict[str, object] = {}
        for path, name in zip(paths, names, strict=True):
            raw = Path(path).read_bytes()
            straight, arcs = truth_by_png[hashlib.sha256(raw).hexdigest()]
            stem = name.removesuffix(".png")
            payload[f"{stem}_straight_action_count_lower"] = straight
            payload[f"{stem}_straight_action_count_upper"] = straight
            payload[f"{stem}_arc_action_count_lower"] = arcs
            payload[f"{stem}_arc_action_count_upper"] = arcs
            payload[f"{stem}_error_code"] = "none"
        calls.append(tuple(names))
        return CodexStructuredResult(
            payload,
            _receipt(prompt, paths, names, schema, payload),
        )

    schema = action_count_batch_output_schema()
    assert len(schema["properties"]) == 70
    assert all(task_id not in action_count_batch_prompt() for task_id in TASKS)
    output = tmp_path / "output"
    completed = run_action_count_phase(
        phase="fit",
        dataset_root=dataset,
        action_program_file=action_file,
        plan_file=plan_file,
        output_root=output,
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        executable="/private/synthetic-codex",
        launcher_sha256=LAUNCHER_DIGEST,
        workers=2,
        expected_plan_digest=plan["record_digest"],
        runtime_override=_runtime(),
        underlying_transport=transport,
    )
    assert len(calls) == len(TASKS) == 2
    result = completed["result"]
    assert result["panel_count"] == 28
    assert result["straight"]["exact_rate"] == [28, 28]
    assert result["straight"]["coverage_rate"] == [28, 28]
    assert result["arc"]["exact_rate"] == [28, 28]
    assert completed["cold_replay"]["model_calls_during_replay"] == 0
    phase_root = output / "fit"
    assert (phase_root / "predictions.json").stat().st_mtime_ns <= (
        phase_root / "label_release.json"
    ).stat().st_mtime_ns
    predictions = json.loads((phase_root / "predictions.json").read_bytes())
    assert predictions["individual_action_labels_opened"] is False
    assert all(
        task["journal_terminal"]["terminal_status"] == "success"
        for task in predictions["task_predictions"]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exactly-once replay attempted a third call")

    repeated = run_action_count_phase(
        phase="fit",
        dataset_root=dataset,
        action_program_file=action_file,
        plan_file=plan_file,
        output_root=output,
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        executable="/private/synthetic-codex",
        launcher_sha256=LAUNCHER_DIGEST,
        workers=2,
        expected_plan_digest=plan["record_digest"],
        runtime_override=_runtime(),
        underlying_transport=forbidden,
    )
    assert repeated == completed
    assert len(calls) == 2
