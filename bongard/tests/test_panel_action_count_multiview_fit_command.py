"""Synthetic custody tests for the FIT-only multiview action observer."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_action_count_multiview_adapter import (
    build_action_count_multiview,
    multiview_algorithm_record,
)
from bongard.panel_action_count_multiview_fit_command import (
    VIEW_NAMES,
    multiview_action_count_output_schema,
    multiview_action_count_prompt,
    run_multiview_action_count_fit,
)
from bongard.panel_action_count_phase_command import PLAN_SCHEMA, _truth_records
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASKS = ("hd_action_multiview_fixture_a_0000", "hd_action_multiview_fixture_b_0000")
STYLES = ("circle", "normal", "square", "triangle", "zigzag")


def _png(seed: int) -> bytes:
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    inset = 110 + seed % 40
    draw.polygon(
        [(inset, 390), (256, 80 + seed % 40), (410 - seed % 30, 390)],
        outline="black",
        width=5,
    )
    draw.arc((140, 130, 380, 410), seed % 30, 180 + seed % 30, fill="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


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
    truth_by_source: dict[str, tuple[int, int]] = {}
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
                path = dataset / "hd/images" / task_id / str(folder) / f"{panel_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                truth_by_source[hashlib.sha256(payload).hexdigest()] = (straight, arcs)
            sides.append(side)
        programs[task_id] = sides
    action_file = dataset / "hd/hd_action_programs.json"
    action_file.parent.mkdir(parents=True, exist_ok=True)
    action_raw = canonical_json(programs) + b"\n"
    action_file.write_bytes(action_raw)
    truth = _truth_records(programs, TASKS)
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
    return dataset, action_file, plan_file, plan, truth_by_source


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


def test_multiview_adapter_is_exact_and_binds_source_and_every_view() -> None:
    source = _png(7)
    first_png, first = build_action_count_multiview(source)
    second_png, second = build_action_count_multiview(source)
    assert first_png == second_png
    assert first == second
    assert first["source_png_sha256"] == hashlib.sha256(source).hexdigest()
    assert first["montage_png_sha256"] == hashlib.sha256(first_png).hexdigest()
    assert first["algorithm_record_digest"] == multiview_algorithm_record()["record_digest"]
    assert first["all_views_derived_only_from_source_png"] is True
    assert first["semantic_action_count_inferred"] is False
    with Image.open(BytesIO(first_png)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (1024, 1024)


def test_fit_multiview_batches_predictions_before_labels_and_cold_replays(
    tmp_path: Path,
) -> None:
    dataset, action_file, plan_file, plan, truth_by_source = _fixture(tmp_path)
    truth_by_montage: dict[str, tuple[int, int]] = {}
    for path in sorted((dataset / "hd/images").glob("*/*/*.png")):
        source = path.read_bytes()
        montage, _record = build_action_count_multiview(source)
        truth_by_montage[hashlib.sha256(montage).hexdigest()] = truth_by_source[
            hashlib.sha256(source).hexdigest()
        ]
    calls: list[tuple[str, ...]] = []

    def transport(prompt, paths, names, schema, **_kwargs):
        assert tuple(names) == VIEW_NAMES
        assert len(paths) == 14
        assert not any(task_id in prompt for task_id in TASKS)
        assert "bottom-right is a blurred coarse carrier-density view" in prompt
        payload: dict[str, object] = {}
        for path, name in zip(paths, names, strict=True):
            raw = Path(path).read_bytes()
            with Image.open(BytesIO(raw)) as image:
                assert image.size == (1024, 1024)
            straight, arcs = truth_by_montage[hashlib.sha256(raw).hexdigest()]
            stem = name.removesuffix(".png")
            for axis, truth in (("straight", straight), ("arc", arcs)):
                payload[f"{stem}_{axis}_best_count"] = truth
                for slot in range(1, 4):
                    payload[f"{stem}_{axis}_alternative_{slot}"] = 10
                payload[f"{stem}_{axis}_count_lower"] = truth
                payload[f"{stem}_{axis}_count_upper"] = truth
            payload[f"{stem}_error_code"] = "none"
        calls.append(tuple(names))
        return CodexStructuredResult(payload, _receipt(prompt, paths, names, schema, payload))

    schema = multiview_action_count_output_schema()
    assert len(schema["properties"]) == 182
    assert all(task_id not in multiview_action_count_prompt() for task_id in TASKS)
    output = tmp_path / "output"
    completed = run_multiview_action_count_fit(
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
    for axis in ("straight", "arc"):
        assert result[axis]["top1_exact_rate"] == [28, 28]
        assert result[axis]["finite_candidate_set_coverage_rate"] == [28, 28]
        assert result[axis]["fallback_interval_coverage_rate"] == [28, 28]
        assert result[axis]["task_max_residual_curves"]["top1"][
            "minimum_zero_omission_radius"
        ] == 0
    assert completed["cold_replay"]["model_calls_during_replay"] == 0
    assert completed["cold_replay"]["all_raw_and_multiview_bytes_rebuilt"] is True
    phase_root = output / "fit"
    assert (phase_root / "predictions.json").stat().st_mtime_ns <= (
        phase_root / "label_release.json"
    ).stat().st_mtime_ns
    predictions = json.loads((phase_root / "predictions.json").read_bytes())
    assert predictions["individual_action_labels_opened_by_process"] is False
    assert len(predictions["external_journal_terminal_records"]) == 2
    assert all(
        terminal["terminal_status"] == "success"
        for terminal in predictions["external_journal_terminal_records"]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exactly-once replay attempted another call")

    repeated = run_multiview_action_count_fit(
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

