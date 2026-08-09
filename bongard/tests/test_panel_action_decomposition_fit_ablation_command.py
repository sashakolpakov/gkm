"""Custody and total-parser tests for the bounded FIT decomposition ablation."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_action_count_phase_command import PLAN_SCHEMA, _truth_records
from bongard.panel_action_decomposition_fit_ablation_command import (
    COMPONENTS,
    DEFAULT_PARENT_OUTCOME,
    SELECTED_TASK_IDS,
    VIEW_NAMES,
    _parse_payload,
    action_decomposition_output_schema,
    action_decomposition_prompt,
    run_action_decomposition_fit_ablation,
)
from bongard.panel_action_decomposition_threeview_adapter import (
    build_action_decomposition_threeview,
    threeview_algorithm_record,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _receipt,
)
from bongard.transport import CodexStructuredResult, validate_codex_strict_output_schema


STYLES = ("circle", "normal", "square", "triangle", "zigzag")


def _png(seed: int) -> bytes:
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    inset = 100 + seed % 45
    draw.polygon(
        [(inset, 390), (256, 75 + seed % 50), (415 - seed % 30, 390)],
        outline="black",
        width=5,
    )
    draw.arc((125, 115, 390, 420), seed % 35, 175 + seed % 35, fill="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _actions(
    normal_straight: int,
    decorated_straight: int,
    normal_arc: int,
    decorated_arc: int,
    offset: int,
) -> list[str]:
    actions = [f"line_normal_0.100-0.200" for _ in range(normal_straight)]
    actions.extend(
        f"line_{STYLES[(offset + index) % 4 * 2 % 5]}_0.100-0.200"
        for index in range(decorated_straight)
    )
    # Guard against the modular expression selecting normal.
    actions = [item.replace("line_normal", "line_triangle") if index >= normal_straight else item for index, item in enumerate(actions)]
    actions.extend("arc_normal_0.100-0.200" for _ in range(normal_arc))
    decorated_styles = ("circle", "square", "triangle", "zigzag")
    actions.extend(
        f"arc_{decorated_styles[(offset + index) % 4]}_0.100-0.200"
        for index in range(decorated_arc)
    )
    return actions


def _fixture(tmp_path: Path):
    dataset = tmp_path / "ShapeBongard_V2"
    programs: dict[str, object] = {}
    truth_by_source: dict[str, tuple[int, int, int, int]] = {}
    seed = 0
    for task_offset, task_id in enumerate(SELECTED_TASK_IDS):
        sides: list[object] = []
        for side_index, folder in enumerate((1, 0)):
            side: list[object] = []
            for panel_index in range(7):
                normal_straight = (panel_index + task_offset) % 4
                decorated_straight = (side_index + panel_index) % 3
                normal_arc = (panel_index + side_index) % 2
                decorated_arc = (task_offset + panel_index) % 2
                side.append(
                    [[
                        *_actions(
                            normal_straight,
                            decorated_straight,
                            normal_arc,
                            decorated_arc,
                            seed,
                        )
                    ]]
                )
                payload = _png(seed)
                seed += 1
                path = dataset / "hd/images" / task_id / str(folder) / f"{panel_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                truth_by_source[hashlib.sha256(payload).hexdigest()] = (
                    normal_straight,
                    decorated_straight,
                    normal_arc,
                    decorated_arc,
                )
            sides.append(side)
        programs[task_id] = sides
    action_file = dataset / "hd/hd_action_programs.json"
    action_file.parent.mkdir(parents=True, exist_ok=True)
    action_raw = canonical_json(programs) + b"\n"
    action_file.write_bytes(action_raw)
    truth = _truth_records(programs, SELECTED_TASK_IDS)
    plan = {
        "schema": PLAN_SCHEMA,
        "current_state": {"selected_panel_pixels_read": False, "model_calls_made": 0},
        "calibration_authority": {
            "python_is_canonical_authority": True,
            "lean_required": False,
        },
        "dataset_bindings": {
            "hd_action_program_raw_sha256": "sha256:" + hashlib.sha256(action_raw).hexdigest(),
        },
        "cohorts": {"fit": {"task_ids": list(SELECTED_TASK_IDS)}},
    }
    plan["record_digest"] = "sha256:" + canonical_digest(plan)
    plan_file = tmp_path / "plan.json"
    plan_file.write_bytes(canonical_json(plan) + b"\n")
    return (
        dataset,
        action_file,
        plan_file,
        plan,
        truth_by_source,
        "sha256:" + canonical_digest(truth),
    )


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


def _presentation() -> list[dict[str, object]]:
    return [
        {
            "model_visible_name": name,
            "panel_id": f"hd/fixture/1/{index}.png",
            "source_png_sha256": f"{index:064x}",
            "threeview_record": {
                "record_digest": "sha256:" + f"{index + 20:064x}",
                "montage_png_sha256": f"{index + 40:064x}",
            },
        }
        for index, name in enumerate(VIEW_NAMES)
    ]


def _valid_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    for name in VIEW_NAMES:
        stem = name.removesuffix(".png")
        payload[f"{stem}_decomposition_counts"] = [2, 1, 1, 0]
        payload[f"{stem}_error_code"] = "none"
    return payload


def test_threeview_adapter_blanks_density_quadrant_and_binds_pillow() -> None:
    source = _png(4)
    first_png, first = build_action_decomposition_threeview(source)
    second_png, second = build_action_decomposition_threeview(source)
    assert first_png == second_png
    assert first == second
    assert first["coarse_carrier_density_generated"] is False
    assert threeview_algorithm_record()["pillow_version"]
    with Image.open(BytesIO(first_png)) as image:
        assert image.size == (1024, 1024)
        assert image.crop((512, 512, 1024, 1024)).getextrema() == (
            (255, 255),
            (255, 255),
            (255, 255),
        )


def test_schema_bounds_every_flat_count_and_prompt_pins_flattening_order() -> None:
    schema = action_decomposition_output_schema()
    validate_codex_strict_output_schema(schema)
    assert len(schema["properties"]) == 28
    arrays = [
        value for name, value in schema["properties"].items()
        if name.endswith("_decomposition_counts")
    ]
    assert len(arrays) == 14
    assert all(value["items"] == {"type": "integer", "enum": list(range(10))} for value in arrays)
    prompt = action_decomposition_prompt()
    assert ", ".join(COMPONENTS) in prompt
    assert "length exactly 4, 8, 12, or 16" in prompt


@pytest.mark.parametrize("bad_length", (1, 5, 20))
def test_schema_valid_bad_length_is_one_panel_error_not_parser_failure(
    bad_length: int,
) -> None:
    payload = _valid_payload()
    payload["view_03_decomposition_counts"] = [0] * bad_length
    rows = _parse_payload(payload, _presentation())
    assert len(rows) == 14
    assert rows[3]["error_code"] == "invalid_tuple_array_length"
    assert rows[3]["raw_decomposition_counts"] == [0] * bad_length
    assert all(row["error_code"] == "none" for index, row in enumerate(rows) if index != 3)


def test_unreadable_nonempty_is_one_panel_error_and_receipt_payload_is_preserved() -> None:
    payload = _valid_payload()
    payload["view_09_error_code"] = "unreadable"
    rows = _parse_payload(payload, _presentation())
    assert rows[9]["error_code"] == "invalid_unreadable_payload"
    assert rows[9]["raw_decomposition_counts"] == [2, 1, 1, 0]
    assert sum(row["error_code"] == "none" for row in rows) == 13


def test_fit_ablation_batches_four_tasks_projects_totals_and_cold_replays(
    tmp_path: Path,
) -> None:
    dataset, action_file, plan_file, plan, truth_by_source, truth_digest = _fixture(tmp_path)
    truth_by_montage: dict[str, tuple[int, int, int, int]] = {}
    for path in sorted((dataset / "hd/images").glob("*/*/*.png")):
        source = path.read_bytes()
        montage, _record = build_action_decomposition_threeview(source)
        truth_by_montage[hashlib.sha256(montage).hexdigest()] = truth_by_source[
            hashlib.sha256(source).hexdigest()
        ]
    calls: list[tuple[str, ...]] = []

    def transport(prompt, paths, names, schema, **_kwargs):
        assert tuple(names) == VIEW_NAMES
        assert len(paths) == 14
        assert not any(task_id in prompt for task_id in SELECTED_TASK_IDS)
        payload: dict[str, object] = {}
        for path, name in zip(paths, names, strict=True):
            values = truth_by_montage[hashlib.sha256(Path(path).read_bytes()).hexdigest()]
            stem = name.removesuffix(".png")
            payload[f"{stem}_decomposition_counts"] = list(values)
            payload[f"{stem}_error_code"] = "none"
        calls.append(tuple(names))
        return CodexStructuredResult(payload, _receipt(prompt, paths, names, schema, payload))

    output = tmp_path / "output"
    completed = run_action_decomposition_fit_ablation(
        dataset_root=dataset,
        action_program_file=action_file,
        plan_file=plan_file,
        parent_outcome_file=DEFAULT_PARENT_OUTCOME,
        output_root=output,
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        executable="/private/synthetic-codex",
        launcher_sha256=LAUNCHER_DIGEST,
        workers=4,
        expected_plan_digest=plan["record_digest"],
        expected_selected_truth_manifest_digest=truth_digest,
        runtime_override=_runtime(),
        underlying_transport=transport,
    )
    assert len(calls) == 4
    result = completed["result"]
    assert result["panel_count"] == 56
    assert result["straight"]["finite_candidate_set_coverage_rate"] == [56, 56]
    assert result["arc"]["finite_candidate_set_coverage_rate"] == [56, 56]
    assert result["decomposition"]["finite_decomposition_set_coverage_rate"] == [56, 56]
    assert result["decomposition"]["joint_total_pair_coverage_rate"] == [56, 56]
    assert completed["cold_replay"]["model_calls_during_replay"] == 0
    phase_root = output / "fit_ablation"
    assert (phase_root / "predictions.json").stat().st_mtime_ns <= (
        phase_root / "label_release.json"
    ).stat().st_mtime_ns

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exactly-once replay attempted another call")

    repeated = run_action_decomposition_fit_ablation(
        dataset_root=dataset,
        action_program_file=action_file,
        plan_file=plan_file,
        parent_outcome_file=DEFAULT_PARENT_OUTCOME,
        output_root=output,
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        executable="/private/synthetic-codex",
        launcher_sha256=LAUNCHER_DIGEST,
        workers=4,
        expected_plan_digest=plan["record_digest"],
        expected_selected_truth_manifest_digest=truth_digest,
        runtime_override=_runtime(),
        underlying_transport=forbidden,
    )
    assert repeated == completed
    assert len(calls) == 4

