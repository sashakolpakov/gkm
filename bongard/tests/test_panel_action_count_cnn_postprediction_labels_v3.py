from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
import bongard.panel_action_count_cnn_postprediction_labels_v3 as authority
from bongard.panel_action_count_cnn_postprediction_labels_v3 import (
    CatalogTarget,
    LabelAuthorityBindings,
    LabelSources,
    POSTPREDICTION_LABEL_CONTRACT,
    PREDICTION_SCHEMA,
    PostPredictionLabelError,
    PredictionBarrier,
    derive_labels_after_durable_predictions,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
PANEL_ID = "hd/hd_synthetic_0000/1/0.png"


def _prediction(*, panel_id: str = PANEL_ID) -> dict[str, object]:
    body: dict[str, object] = {
        "catalog_class_order": ["catalog_unresolved", "nonconvex", "convex"],
        "checkpoint_state_dict_sha256": SHA_C,
        "config_digest": SHA_D,
        "joint_q": None,
        "joint_q_record_digest": None,
        "panel_ids": [panel_id],
        "panel_manifest_record_digest": SHA_B,
        "plan_record_digest": SHA_A,
        "rows": [
            {
                "arc_probabilities": [1.0] + [0.0] * 9,
                "catalog_probabilities": [0.0, 0.0, 1.0],
                "panel_id": panel_id,
                "straight_probabilities": [0.0] * 4 + [1.0] + [0.0] * 5,
            }
        ],
        "schema": PREDICTION_SCHEMA,
        "stage": "calibration",
        "straight_class_order": list(range(10)),
        "arc_class_order": list(range(10)),
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _write_prediction(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _programs() -> dict[str, object]:
    empty_side = [[["arc_normal_0_90_1"]] for _ in range(7)]
    selected_side = [[["line_normal_0_1"] * 4] for _ in range(7)]
    return {"hd_synthetic_0000": [selected_side, empty_side]}


def _program_bytes() -> bytes:
    return canonical_json(_programs()) + b"\n"


def _label_bindings() -> LabelAuthorityBindings:
    return LabelAuthorityBindings(
        hd_action_program_raw_sha256="sha256:"
        + hashlib.sha256(_program_bytes()).hexdigest(),
        catalog_algorithm_digest="sha256:" + "1" * 64,
        catalog_audit_record_digest="sha256:" + "2" * 64,
        catalog_authority_source_sha256="sha256:" + "3" * 64,
    )


def test_label_source_loader_runs_only_after_valid_durable_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    prediction_path = tmp_path / "prediction.json"
    _write_prediction(prediction_path, _prediction())
    observed: list[PredictionBarrier] = []
    events: list[str] = []
    original_fsync = authority.os.fsync

    def tracked_fsync(descriptor: int) -> None:
        original_fsync(descriptor)
        events.append("fsync")

    monkeypatch.setattr(authority.os, "fsync", tracked_fsync)

    def load_sources(barrier: PredictionBarrier) -> LabelSources:
        assert events == ["fsync", "fsync"]
        events.append("label_source_loader")
        observed.append(barrier)
        assert barrier.protocol == "fsync-file-and-parent-then-byte-identical-reload/v3"
        return LabelSources(
            hd_action_program_raw=_program_bytes(),
            catalog_lookup=lambda _actions: CatalogTarget(
                raw_target=1,
                supervised_class="convex",
                match_kind="synthetic_exact_signature",
            ),
            authority_bindings=_label_bindings(),
        )

    rows = derive_labels_after_durable_predictions(
        prediction_path=prediction_path,
        expected_stage="calibration",
        expected_panel_ids=[PANEL_ID],
        expected_plan_record_digest=SHA_A,
        expected_panel_manifest_record_digest=SHA_B,
        expected_checkpoint_state_dict_sha256=SHA_C,
        expected_config_digest=SHA_D,
        expected_label_authority_bindings=_label_bindings(),
        source_loader=load_sources,
    )
    assert len(observed) == 1
    assert events == ["fsync", "fsync", "label_source_loader"]
    assert rows == (
        {
            "arc_action_count": 0,
            "catalog_convexity_class": "convex",
            "catalog_convexity_target": 1,
            "catalog_match_kind": "synthetic_exact_signature",
            "panel_id": PANEL_ID,
            "straight_action_count": 4,
        },
    )


def test_invalid_or_incomplete_prediction_cannot_open_label_sources(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.json"
    value = _prediction(panel_id="hd/hd_wrong_0000/1/0.png")
    _write_prediction(prediction_path, value)
    opened = False

    def forbidden_loader(_barrier: PredictionBarrier) -> LabelSources:
        nonlocal opened
        opened = True
        raise AssertionError("label sources opened before a valid prediction barrier")

    with pytest.raises(PostPredictionLabelError, match="panel order differs"):
        derive_labels_after_durable_predictions(
            prediction_path=prediction_path,
            expected_stage="calibration",
            expected_panel_ids=[PANEL_ID],
            expected_plan_record_digest=SHA_A,
            expected_panel_manifest_record_digest=SHA_B,
            expected_checkpoint_state_dict_sha256=SHA_C,
            expected_config_digest=SHA_D,
            expected_label_authority_bindings=_label_bindings(),
            source_loader=forbidden_loader,
        )
    assert opened is False


def test_postprediction_module_import_has_no_file_read_side_effect() -> None:
    source = Path(
        "bongard/panel_action_count_cnn_postprediction_labels_v3.py"
    ).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Expr, ast.Assign)):
            assert not any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"read_bytes", "read_text", "open"}
                for child in ast.walk(node)
            )


def test_catalog_projection_is_scoped_to_catalog_axis_and_unresolved_is_gap() -> None:
    projection = POSTPREDICTION_LABEL_CONTRACT["catalog_typed_projection"]
    assert projection == {
        "axis": "catalog_convexity",
        "singleton_class_1": "catalog_nonconvex",
        "singleton_class_2": "catalog_convex",
        "any_set_containing_class_0_catalog_unresolved": "whole-axis-GAP",
        "geometric_turning_axis_used": False,
        "not_applicable_used_for_catalog_unresolved": False,
    }
