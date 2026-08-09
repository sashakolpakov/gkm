"""Replay the four-phase, three-head action-count CNN v2 preregistration."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.panel_action_count_cnn_preregister_v2 import (
    CALIBRATION_ORDER_STATISTIC,
    build_v2_preregistration,
)


BONGARD = Path(__file__).resolve().parents[1]
ROOT = BONGARD.parent
CORPUS = ROOT / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
DATA = BONGARD / "data"
PLAN = DATA / "panel_action_count_cnn_preregistration_20260810_v2.json"
DEVELOPMENT = DATA / "panel_action_count_cnn_development_labels_20260810_v2.json"
CAL_PANELS = DATA / "panel_action_count_cnn_calibration_panels_20260810_v2.json"
CAL_LABELS = DATA / "panel_action_count_cnn_calibration_labels_sealed_20260810_v2.json"
EVAL_PANELS = DATA / "panel_action_count_cnn_evaluation_panels_20260810_v2.json"
EVAL_LABELS = DATA / "panel_action_count_cnn_evaluation_labels_sealed_20260810_v2.json"
LEDGER = (
    ROOT
    / "downloads/ShapeBongard_V2_full/panel_soft_exact_unused_train_20260809_ranked_v1"
    / "research-exposure-successors"
    / "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
)


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json(value) + b"\n"
    body = dict(value)
    digest = body.pop("record_digest")
    assert digest == "sha256:" + canonical_digest(body)
    return value


def _rebuild() -> tuple[dict[str, object], ...]:
    return build_v2_preregistration(
        repository_root=ROOT,
        dataset_root=CORPUS,
        authority_source_path=BONGARD / "panel_action_count_cnn_preregister_v2.py",
        v1_plan_path=DATA / "panel_action_count_cnn_preregistration_20260810_v1.json",
        v1_development_path=DATA
        / "panel_action_count_cnn_development_labels_20260810_v1.json",
        v1_evaluation_panels_path=DATA
        / "panel_action_count_cnn_evaluation_panels_20260810_v1.json",
        v1_evaluation_labels_path=DATA
        / "panel_action_count_cnn_evaluation_labels_sealed_20260810_v1.json",
        action_count_plan_path=DATA
        / "panel_action_count_calibration_preregistration_20260809_v1.json",
        family_plan_path=DATA
        / "panel_convex_four_lines_same_family_train_drill_20260809_v1.json",
        historical_exposure_path=DATA / "historical_exposure_v1.json",
        cumulative_exposure_ledger_path=LEDGER,
        catalog_audit_path=DATA / "panel_convexity_catalog_audit_20260810_v1.json",
        shape_rows_path=ROOT / "downloads/Bongard-LOGO/data/human_designed_shapes.tsv",
        attribute_rows_path=ROOT
        / "downloads/Bongard-LOGO/data/human_designed_shapes_attributes.tsv",
        hd_programs_path=CORPUS / "hd/hd_action_programs.json",
        bd_programs_path=CORPUS / "bd/bd_action_programs.json",
        split_path=CORPUS / "ShapeBongard_V2_split.json",
        release_descriptor_path=DATA / "shape_bongard_v2_release_v1.json",
        development_output_path=DEVELOPMENT,
        calibration_panels_output_path=CAL_PANELS,
        calibration_labels_output_path=CAL_LABELS,
        evaluation_panels_output_path=EVAL_PANELS,
        evaluation_labels_output_path=EVAL_LABELS,
    )


def test_v2_replays_exactly_without_opening_any_png(monkeypatch) -> None:
    archived = tuple(
        _record(path)
        for path in (PLAN, DEVELOPMENT, CAL_PANELS, CAL_LABELS, EVAL_PANELS, EVAL_LABELS)
    )
    assert [value["record_digest"] for value in archived] == [
        "sha256:0de57e610763a7fb77adbcaeb2be21b20864a02eb5af0656b76c291ef5b0a3a8",
        "sha256:c72d09eaa2bee02572694dacdb48ec80d2e23615c1c54f4c6616136b235b3d52",
        "sha256:17f1291297545f573727a03bd49f64dc92e418a12586a29f53509a3373554f24",
        "sha256:3a057ec3fbc05991098579430682e2e110a1d829f036484cfff2ab54b76e11d4",
        "sha256:149bc75c1a5f39e7fcbe3f6b599a3d0bfc3ed04b5727d22c5ed6917d24c32b6e",
        "sha256:a15e019391e1dd3f6788b23630f0f5e8583f0a8930ec8f7dd2b07e0e84b8076f",
    ]
    original = Path.read_bytes
    opened: list[Path] = []

    def guard(path: Path) -> bytes:
        if path.suffix.lower() == ".png":
            opened.append(path)
            raise AssertionError("v2 metadata authority opened a PNG")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    assert _rebuild() == archived
    assert not opened


def test_v2_cohorts_are_exact_unused_disjoint_and_permanently_tainted() -> None:
    plan = _record(PLAN)
    selected: list[str] = []
    for name, count in (
        ("train", 800),
        ("validation", 100),
        ("calibration", 100),
        ("evaluation", 100),
    ):
        cohort = plan["cohorts"][name]
        assert cohort["task_count"] == count
        assert cohort["panel_count"] == count * 14
        selected.extend(cohort["task_ids"])
    assert len(selected) == len(set(selected)) == 1100
    assert all(
        "convex" not in task_id and "has_four_straight_lines" not in task_id
        for task_id in selected
    )
    ledger = ExposureLedger.from_dict(json.loads(LEDGER.read_bytes()))
    exposed = {task for event in ledger.events for task in event.task_ids}
    assert not set(selected).intersection(exposed)
    v1 = _record(DATA / "panel_action_count_cnn_preregistration_20260810_v1.json")
    assert selected[:1000] == v1["oracle_taint_record"]["selected_task_ids"]
    assert plan["oracle_taint_record"]["selected_task_ids"] == selected
    assert plan["oracle_taint_record"]["selected_panel_count"] == 15_400
    assert plan["current_state"]["selected_panel_png_bytes_read"] == 0
    assert plan["current_state"]["calibration_panel_png_bytes_read"] == 0
    assert plan["current_state"]["evaluation_panel_png_bytes_read"] == 0


def test_v2_catalog_labels_custody_calibration_and_claim_limits_are_closed() -> None:
    plan = _record(PLAN)
    development = _record(DEVELOPMENT)
    cal_panels = _record(CAL_PANELS)
    cal_labels = _record(CAL_LABELS)
    eval_panels = _record(EVAL_PANELS)
    eval_labels = _record(EVAL_LABELS)
    assert cal_panels["panel_ids"] == [row["panel_id"] for row in cal_labels["rows"]]
    assert eval_panels["panel_ids"] == [row["panel_id"] for row in eval_labels["rows"]]
    all_rows = (
        development["cohorts"]["train"]["rows"]
        + development["cohorts"]["validation"]["rows"]
        + cal_labels["rows"]
        + eval_labels["rows"]
    )
    assert len(all_rows) == 15_400
    assert {row["catalog_convexity_class"] for row in all_rows} == {
        "catalog_unresolved",
        "convex",
        "nonconvex",
    }
    assert all(row["catalog_convexity_target"] in {-1, 0, 1} for row in all_rows)
    assert all(
        row["catalog_match_kind"]
        in {"direct_exact_signature", "bd_singleton_compatibility_alias"}
        for row in all_rows
    )
    bindings = plan["dataset_and_authority_bindings"]
    assert bindings["catalog_direct_signature_count"] == 627
    assert bindings["catalog_compatibility_alias_count"] == 4
    assert bindings["catalog_authority_source_sha256"] == (
        "0652589108c0f77ec86550a84eeeac3e65edffb81969c61fc9705571cb1c5286"
    )
    calibration = plan["calibration_protocol"]
    assert CALIBRATION_ORDER_STATISTIC == 96
    assert calibration["order_statistic_one_indexed"] == 96
    assert calibration["canonical_deployment_q"] == "joint_q_only"
    assert calibration["zero_miss_max_used"] is False
    limits = plan["formal_claim_limits"]
    assert limits["conformal_grant_formally_transfers_to_target_family"] is False
    assert limits["target_exclusions_may_be_called_certified"] is False
    training = plan["training_protocol"]
    assert "source_png_sha256" in training["augmentation_and_shuffle_key"]
    assert "panel_id" in training["augmentation_and_shuffle_key"]
    assert training["staged_pixel_custody"] == {
        "calibration_and_evaluation_bytes_may_not_be_hashed_stat-ed_or_decoded_in_train-validation_stage": True,
        "calibration_tensors_not_decoded_before_validation_gate_pass": True,
        "evaluation_precommit_panel_count_after_q_freeze": 1400,
        "evaluation_tensors_not_decoded_before_joint_q_freeze": True,
        "calibration_precommit_panel_count_after_validation_pass": 1400,
        "train-validation_precommit_panel_count": 12600,
    }
    assert "catalog_unresolved" in training["catalog_unresolved_downstream_rule"]
    assert plan["transport_and_language_limits"]["lean_present"] is False

