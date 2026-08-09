"""Cold replay for the metadata-only supervised action-count CNN split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.panel_action_count_cnn_preregister import build_preregistration


BONGARD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BONGARD_ROOT.parent
DATASET_ROOT = REPOSITORY_ROOT / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
PLAN_PATH = BONGARD_ROOT / "data/panel_action_count_cnn_preregistration_20260810_v1.json"
DEVELOPMENT_PATH = (
    BONGARD_ROOT / "data/panel_action_count_cnn_development_labels_20260810_v1.json"
)
EVALUATION_PANELS_PATH = (
    BONGARD_ROOT / "data/panel_action_count_cnn_evaluation_panels_20260810_v1.json"
)
EVALUATION_LABELS_PATH = (
    BONGARD_ROOT
    / "data/panel_action_count_cnn_evaluation_labels_sealed_20260810_v1.json"
)
CUMULATIVE_EXPOSURE_PATH = (
    REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/panel_soft_exact_unused_train_20260809_ranked_v1"
    / "research-exposure-successors"
    / "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
)


def _assert_canonical_record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    record = json.loads(raw)
    assert raw == canonical_json(record) + b"\n"
    body = dict(record)
    digest = body.pop("record_digest")
    assert digest == "sha256:" + canonical_digest(body)
    return record


def test_supervised_action_count_preregistration_cold_replays_without_pngs(
    monkeypatch,
) -> None:
    archived = (
        _assert_canonical_record(PLAN_PATH),
        _assert_canonical_record(DEVELOPMENT_PATH),
        _assert_canonical_record(EVALUATION_PANELS_PATH),
        _assert_canonical_record(EVALUATION_LABELS_PATH),
    )
    assert [item["record_digest"] for item in archived] == [
        "sha256:bbb8d380dc291a80d2fb89ee522c5ad6b5be2f0d67e5a2a80fc9907cb3337777",
        "sha256:b37478bcfeb2b20957eca446b40e26eb3fef5bc285a4d24c1176099dd32f181a",
        "sha256:68c21f501be6bd7f92e9be75eda3085c498317efe037adafebfc809e1f8ea2fd",
        "sha256:0df53632a4395324196c9902a5e4428cf908f4787bd2abcf82173802dc645da4",
    ]

    original_read_bytes = Path.read_bytes
    opened_pngs: list[Path] = []

    def guarded_read_bytes(path: Path) -> bytes:
        if path.suffix.lower() == ".png":
            opened_pngs.append(path)
            raise AssertionError("metadata preregistration attempted to open a PNG")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    rebuilt = build_preregistration(
        repository_root=REPOSITORY_ROOT,
        dataset_root=DATASET_ROOT,
        authority_source_path=BONGARD_ROOT / "panel_action_count_cnn_preregister.py",
        development_label_manifest_path=DEVELOPMENT_PATH,
        evaluation_panel_manifest_path=EVALUATION_PANELS_PATH,
        evaluation_label_manifest_path=EVALUATION_LABELS_PATH,
        action_count_plan_path=BONGARD_ROOT
        / "data/panel_action_count_calibration_preregistration_20260809_v1.json",
        family_plan_path=BONGARD_ROOT
        / "data/panel_convex_four_lines_same_family_train_drill_20260809_v1.json",
        historical_exposure_path=BONGARD_ROOT / "data/historical_exposure_v1.json",
        cumulative_exposure_ledger_path=CUMULATIVE_EXPOSURE_PATH,
        action_program_audit_path=BONGARD_ROOT
        / "data/shape_bongard_v2_action_program_audit_v1.json",
        release_descriptor_path=BONGARD_ROOT / "data/shape_bongard_v2_release_v1.json",
    )
    assert not opened_pngs
    assert rebuilt == archived


def test_supervised_action_count_split_is_exact_unused_and_disjoint() -> None:
    plan = _assert_canonical_record(PLAN_PATH)
    cohorts = plan["cohorts"]
    train_tasks = cohorts["train"]["task_ids"]
    validation_tasks = cohorts["validation"]["task_ids"]
    evaluation_tasks = cohorts["evaluation"]["task_ids"]
    assert len(train_tasks) == len(set(train_tasks)) == 800
    assert len(validation_tasks) == len(set(validation_tasks)) == 100
    assert len(evaluation_tasks) == len(set(evaluation_tasks)) == 100
    selected = train_tasks + validation_tasks + evaluation_tasks
    assert len(selected) == len(set(selected)) == 1000

    split = SplitIndex.load(DATASET_ROOT / "ShapeBongard_V2_split.json")
    assert all(split.assignment(task_id).split == "train" for task_id in selected)
    assert all(
        "convex" not in task_id and "has_four_straight_lines" not in task_id
        for task_id in selected
    )

    exposure = ExposureLedger.from_dict(json.loads(CUMULATIVE_EXPOSURE_PATH.read_bytes()))
    cumulative_tasks = {
        task_id for event in exposure.events for task_id in event.task_ids
    }
    assert len(exposure.events) == 158
    assert len(cumulative_tasks) == 314
    assert not set(selected).intersection(cumulative_tasks)
    bound = plan["exclusions"]["cumulative_research_exposure_ledger"]
    assert bound["ledger_digest"] == exposure.digest
    assert bound["event_count"] == 158
    assert bound["exposed_task_count"] == 314
    assert bound["exposed_task_ids_digest"] == (
        "sha256:" + canonical_digest(sorted(cumulative_tasks))
    )

    taint = plan["oracle_taint_record"]
    assert taint["permanent"] is True
    assert taint["selected_task_ids"] == selected
    assert taint["selected_task_ids_digest"] == (
        "sha256:" + canonical_digest(selected)
    )
    assert taint["selected_panel_count"] == 14_000


def test_manifests_separate_development_from_sealed_evaluation_labels() -> None:
    plan = _assert_canonical_record(PLAN_PATH)
    development = _assert_canonical_record(DEVELOPMENT_PATH)
    evaluation_panels = _assert_canonical_record(EVALUATION_PANELS_PATH)
    evaluation_labels = _assert_canonical_record(EVALUATION_LABELS_PATH)

    train_rows = development["cohorts"]["train"]["rows"]
    validation_rows = development["cohorts"]["validation"]["rows"]
    eval_panel_ids = evaluation_panels["panel_ids"]
    eval_rows = evaluation_labels["rows"]
    assert len(train_rows) == 11_200
    assert len(validation_rows) == len(eval_panel_ids) == len(eval_rows) == 1_400
    assert eval_panel_ids == [row["panel_id"] for row in eval_rows]
    assert all(isinstance(panel_id, str) for panel_id in eval_panel_ids)
    assert all(not isinstance(panel_id, dict) for panel_id in eval_panel_ids)
    all_rows = train_rows + validation_rows + eval_rows
    panel_ids = [row["panel_id"] for row in all_rows]
    assert len(panel_ids) == len(set(panel_ids)) == 14_000
    assert all(0 <= row["straight_action_count"] <= 9 for row in all_rows)
    assert all(0 <= row["arc_action_count"] <= 9 for row in all_rows)

    assert development["cohorts"]["validation"]["summary"] == {
        "arc_action_count": {
            "0": 690,
            "1": 360,
            "2": 247,
            "3": 18,
            "4": 63,
            "6": 14,
            "8": 8,
        },
        "crossing_task_panel_count": 182,
        "line_decoration_stratum": {
            "decorated_only": 214,
            "mixed_normal_and_decorated": 361,
            "no_straight_actions": 107,
            "normal_only": 718,
        },
        "panel_count": 1400,
        "straight_action_count": {
            "0": 107,
            "1": 23,
            "2": 74,
            "3": 159,
            "4": 188,
            "5": 157,
            "6": 428,
            "7": 160,
            "8": 77,
            "9": 27,
        },
        "straight_count_4_panel_count": 188,
        "thin_task_panel_count": 126,
    }
    assert evaluation_labels["summary"]["straight_count_4_panel_count"] == 226
    assert evaluation_labels["summary"]["thin_task_panel_count"] == 112
    assert evaluation_labels["summary"]["crossing_task_panel_count"] == 210

    for name, path, record in (
        ("development_labels", DEVELOPMENT_PATH, development),
        ("evaluation_panels_label_free", EVALUATION_PANELS_PATH, evaluation_panels),
        ("evaluation_labels_sealed", EVALUATION_LABELS_PATH, evaluation_labels),
    ):
        binding = plan["manifest_bindings"][name]
        raw = path.read_bytes()
        assert binding["record_digest"] == record["record_digest"]
        assert binding["source_sha256"] == (
            "sha256:" + hashlib.sha256(raw).hexdigest()
        )

    state = plan["current_state"]
    assert state["selected_panel_png_bytes_read"] == 0
    assert state["model_training_started"] is False
    assert state["evaluation_labels_opened_by_training_execution"] is False
    assert state["target_family_panel_pixels_read"] is False
    limits = plan["supervision_and_claim_limits"]
    assert limits["all_selected_tasks_and_panels_permanently_oracle_tainted"] is True
    assert limits["pixels_are_the_only_model_inputs"] is True
    assert limits["official_validation_or_test_authorized"] is False
    assert limits["lean_present"] is False
    assert limits["lean_required"] is False

