"""Replay the TRAIN-only decoration-aware action-count preregistration."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.release import load_official_release


BONGARD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BONGARD_ROOT.parent
DATASET_ROOT = REPOSITORY_ROOT / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
PLAN_PATH = BONGARD_ROOT / "data/panel_action_count_calibration_preregistration_20260809_v1.json"
AUDIT_PATH = BONGARD_ROOT / "data/shape_bongard_v2_action_program_audit_v1.json"
STYLES = ("circle", "normal", "square", "triangle", "zigzag")


def _label_records(programs: dict[str, object], tasks: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in tasks:
        sides = programs[task]
        assert isinstance(sides, list) and len(sides) == 2
        for side_index, side in enumerate(sides):
            assert isinstance(side, list) and len(side) == 7
            folder = 1 if side_index == 0 else 0
            for panel_index, panel in enumerate(side):
                assert isinstance(panel, list) and len(panel) == 1
                actions = panel[0]
                assert isinstance(actions, list)
                parsed = [action.split("_", 2)[:2] for action in actions]
                assert all(
                    len(item) == 2
                    and item[0] in {"line", "arc"}
                    and item[1] in STYLES
                    for item in parsed
                )
                lines = [style for kind, style in parsed if kind == "line"]
                arcs = [style for kind, style in parsed if kind == "arc"]
                records.append(
                    {
                        "panel_id": f"hd/{task}/{folder}/{panel_index}.png",
                        "straight_action_count": len(lines),
                        "arc_action_count": len(arcs),
                        "line_action_count_by_style": {
                            style: lines.count(style) for style in STYLES
                        },
                        "arc_action_count_by_style": {
                            style: arcs.count(style) for style in STYLES
                        },
                    }
                )
    return records


def _summary(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counters = {
        name: Counter()
        for name in (
            "panel_count_by_straight_action_count",
            "panel_count_by_arc_action_count",
            "panel_count_by_line_decoration_profile",
            "panel_count_by_arc_presence",
            "panel_count_by_straight_count_and_arc_presence",
            "panel_count_by_straight_count_and_line_decoration_profile",
            "panels_containing_line_style",
            "panels_containing_arc_style",
            "line_action_count_by_style",
            "arc_action_count_by_style",
        )
    }
    for row in records:
        lines = row["straight_action_count"]
        arcs = row["arc_action_count"]
        line_styles = row["line_action_count_by_style"]
        arc_styles = row["arc_action_count_by_style"]
        assert isinstance(lines, int) and isinstance(arcs, int)
        assert isinstance(line_styles, dict) and isinstance(arc_styles, dict)
        normal = line_styles["normal"]
        decorated = lines - normal
        profile = (
            "no_straight_actions"
            if lines == 0
            else "normal_only"
            if decorated == 0
            else "decorated_only"
            if normal == 0
            else "mixed_normal_and_decorated"
        )
        presence = "with_arc" if arcs else "without_arc"
        counters["panel_count_by_straight_action_count"][str(lines)] += 1
        counters["panel_count_by_arc_action_count"][str(arcs)] += 1
        counters["panel_count_by_line_decoration_profile"][profile] += 1
        counters["panel_count_by_arc_presence"][presence] += 1
        counters["panel_count_by_straight_count_and_arc_presence"][
            f"{lines}|{presence}"
        ] += 1
        counters["panel_count_by_straight_count_and_line_decoration_profile"][
            f"{lines}|{profile}"
        ] += 1
        for style, count in line_styles.items():
            counters["line_action_count_by_style"][style] += count
            if count:
                counters["panels_containing_line_style"][style] += 1
        for style, count in arc_styles.items():
            counters["arc_action_count_by_style"][style] += count
            if count:
                counters["panels_containing_arc_style"][style] += 1
    return {name: dict(sorted(counter.items())) for name, counter in counters.items()}


def test_action_count_preregistration_replays_without_pixels_or_model_calls() -> None:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    assert raw == canonical_json(plan) + b"\n"
    body = dict(plan)
    record_digest = body.pop("record_digest")
    assert record_digest == "sha256:" + canonical_digest(body)
    assert record_digest == (
        "sha256:476ff0d602d43ddc6e4c8c6a964939a01c8471936eece71d0caba8a35bda396a"
    )

    split = SplitIndex.load(DATASET_ROOT / "ShapeBongard_V2_split.json")
    action_path = DATASET_ROOT / "hd/hd_action_programs.json"
    action_raw = action_path.read_bytes()
    programs = json.loads(action_raw)
    audit = json.loads(AUDIT_PATH.read_bytes())
    release = load_official_release()
    bindings = plan["dataset_bindings"]
    assert bindings["official_release_descriptor_digest"] == release.digest
    assert bindings["split_source_digest"] == split.source_digest
    assert bindings["split_manifest_digest"] == "sha256:" + canonical_digest(
        split.to_manifest_dict()
    )
    assert bindings["hd_action_program_raw_sha256"] == (
        "sha256:" + hashlib.sha256(action_raw).hexdigest()
    )
    assert bindings["hd_action_program_parsed_canonical_sha256"] == (
        "sha256:" + hashlib.sha256(canonical_json(programs)).hexdigest()
    )
    assert bindings["action_program_audit_digest"] == audit["digest"]

    selection = plan["selection"]
    hd_train = sorted(set(programs).intersection(split.canonical_groups["train"]))
    excluded = sorted(
        task
        for task in hd_train
        if "convex" in task or "has_four_straight_lines" in task
    )
    excluded_set = set(excluded)
    seed = selection["selection_seed"]
    eligible = sorted(
        (task for task in hd_train if task not in excluded_set),
        key=lambda task: (
            hashlib.sha256((seed + "\0" + task).encode()).hexdigest(),
            task,
        ),
    )
    selected = eligible[:60]
    expected_partitions = {
        "fit": selected[:20],
        "calibration": selected[20:40],
        "heldout": selected[40:60],
    }
    assert selection["official_hd_train_task_count"] == 3362 == len(hd_train)
    assert selection["semantic_closure_excluded_train_task_count"] == 454 == len(excluded)
    assert selection["eligible_train_task_count"] == 2908 == len(eligible)
    assert selection["semantic_closure_excluded_train_task_ids_digest"] == (
        "sha256:" + canonical_digest(excluded)
    )
    assert selection["selected_task_ids_digest"] == "sha256:" + canonical_digest(selected)
    assert selection["algorithm_digest"] == "sha256:" + canonical_digest(
        selection["algorithm"]
    )

    all_cohort_tasks: list[str] = []
    for name, expected_tasks in expected_partitions.items():
        cohort = plan["cohorts"][name]
        assert cohort["task_ids"] == expected_tasks
        assert cohort["task_count"] == 20
        assert cohort["task_ids_digest"] == "sha256:" + canonical_digest(expected_tasks)
        assert all(split.assignment(task).split == "train" for task in expected_tasks)
        assert not set(expected_tasks).intersection(excluded)
        records = _label_records(programs, expected_tasks)
        panel_ids = [row["panel_id"] for row in records]
        assert cohort["panel_count"] == len(records) == 280
        assert cohort["panel_ids_digest"] == "sha256:" + canonical_digest(panel_ids)
        assert cohort["action_label_manifest_digest"] == (
            "sha256:" + canonical_digest(records)
        )
        assert cohort["label_summary"] == _summary(records)
        assert set(cohort["label_summary"]["panel_count_by_straight_action_count"]) == {
            str(index) for index in range(10)
        }
        assert set(cohort["label_summary"]["panels_containing_line_style"]) == set(STYLES)
        assert set(cohort["label_summary"]["panels_containing_arc_style"]) == set(STYLES)
        all_cohort_tasks.extend(expected_tasks)
    assert len(all_cohort_tasks) == len(set(all_cohort_tasks)) == 60

    target = plan["target_exclusion"]
    exact_target = [f"hd_convex-has_four_straight_lines_{index:04d}" for index in range(20)]
    assert target["exact_target_family_task_ids"] == exact_target
    assert target["exact_target_family_task_ids_digest"] == (
        "sha256:" + canonical_digest(exact_target)
    )
    assert not set(exact_target).intersection(all_cohort_tasks)
    assert sum(split.assignment(task).split == "train" for task in exact_target) == 18
    assert sum(split.assignment(task).split == "val" for task in exact_target) == 2

    typed = plan["typed_observer_contract"]
    assert typed["output_fields"] == {
        "straight_action_count_lower": "integer_0_to_9",
        "straight_action_count_upper": "integer_0_to_9",
        "arc_action_count_lower": "integer_0_to_9",
        "arc_action_count_upper": "integer_0_to_9",
        "error_code": "null_or_bounded_string",
    }
    assert typed["typed_axes_calibrated_independently"] is True
    assert typed["closure_convexity_gestalt_or_task-concept_output_fields_present"] is False
    assert typed["free_prose_candidate_atoms_present"] is False
    assert typed["formula_synthesis_present"] is False
    assert typed["task_label_prediction_present"] is False

    state = plan["current_state"]
    assert state["selected_panel_pixels_read"] is False
    assert state["target_family_panel_pixels_read"] is False
    assert state["model_calls_made"] == 0
    leakage = plan["leakage_controls"]
    assert leakage["selected_tasks_permanently_oracle_tainted_for_future_bongard_benchmarking"] is True
    assert leakage["official_validation_pixels_authorized"] is False
    assert leakage["official_test_pixels_authorized"] is False
    assert leakage["official_validation_or_test_action_labels_authorized"] is False
    authority = plan["calibration_authority"]
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_present"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
