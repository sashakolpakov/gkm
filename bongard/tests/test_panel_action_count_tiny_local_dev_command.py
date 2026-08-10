"""Focused synthetic tests for the infrastructure-only tiny local observer."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image, ImageDraw

from bongard import panel_action_count_tiny_local_dev_command as subject


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _interval(center: float, radius: float = 0.01) -> dict[str, float]:
    return {
        "center": center,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _targets():
    return (
        {"kind": "line", "line_length": _interval(0.4)},
        {
            "arc_radius": _interval(0.3),
            "arc_sweep_magnitude": _interval(0.5),
            "kind": "arc",
        },
    )


def _png() -> bytes:
    image = Image.new("L", (80, 60), 255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 70, 45), fill=0, width=3)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _authority_panel(*, panel_id: str = "hd/task/0/0.png", disposition="CERTIFIED"):
    base = {
        "action_program_sha256": SHA_A,
        "algorithm_id": "pose-free-action-and-internal-junction-multisets/v1",
        "authority_record_digest": SHA_B,
        "cohort": "train",
        "disposition": disposition,
        "panel_id": panel_id,
        "pixel_instance_assignment": {
            "code": "official_pixel_registration_unavailable",
            "detail": "missing pose",
            "disposition": "GAP",
        },
        "pixel_registration": {
            "code": "official_pixel_registration_unavailable",
            "detail": "missing pose",
            "disposition": "GAP",
        },
        "schema": "gkm.bongard-pose-free-local-action-supervision.v1",
        "sequence_endpoint_localization": {
            "code": "sequence_endpoints_not_visually_identifiable",
            "detail": "unidentifiable",
            "disposition": "GAP",
        },
    }
    if disposition == "GAP":
        return {
            **base,
            "gap": {"code": "unsupported_style", "detail": "x", "disposition": "GAP"},
        }
    return {
        **base,
        "carrier_instance_count": {"disposition": "CERTIFIED", "value": 3},
        "shape_instance_count": {"disposition": "CERTIFIED", "value": 2},
        "shape_multiset": [
            {
                "action_count": 2,
                "action_multiset": [
                    {
                        "length_normalized_micro_interval": {
                            "lower": 399500,
                            "upper": 400500,
                            "unit": "normalized_micro",
                        },
                        "length_source_normalized_milli": 400,
                        "multiplicity": 2,
                        "primitive": "line",
                    }
                ],
                "internal_junction_multiset": [],
                "multiplicity": 1,
            },
            {
                "action_count": 1,
                "action_multiset": [
                    {
                        "multiplicity": 1,
                        "primitive": "arc",
                        "radius_normalized_micro_interval": {
                            "lower": 299500,
                            "upper": 300500,
                            "unit": "normalized_micro",
                        },
                        "radius_source_normalized_milli": 300,
                        "sweep_magnitude_degrees_milli_interval": {
                            "lower": 179640,
                            "upper": 180360,
                            "unit": "degree_milli",
                        },
                        "sweep_magnitude_source_degrees_milli": 180000,
                    }
                ],
                "internal_junction_multiset": [],
                "multiplicity": 1,
            },
        ],
        "supervision_semantics": {},
    }


def _raw_record(*, panel_id: str, line: int, arc: int, catalog: int):
    matrix = [[0.0] * 10 for _ in range(10)]
    matrix[line][arc] = 1.0
    catalog_probabilities = [0.0, 0.0, 0.0]
    catalog_probabilities[catalog] = 1.0
    body = {
        "architecture_id": subject.ARCHITECTURE_ID,
        "catalog_class_order": list(subject.CATALOG_CLASSES),
        "catalog_probabilities": catalog_probabilities,
        "checkpoint_state_dict_sha256": SHA_A,
        "config_digest": SHA_B,
        "joint_count_probabilities_straight_rows_arc_columns": matrix,
        "panel_id": panel_id,
        "pixel_registration_claimed": False,
        "png_sha256": SHA_A,
        "png_size_bytes": 12,
        "schema": subject.RAW_OBSERVATION_SCHEMA,
        "slot_class_order": list(subject.SLOT_CLASSES),
        "slots": [],
        "source_sha256": subject.source_sha256(),
    }
    return subject._seal(body)


def test_model_is_tiny_and_joint_dp_is_exact_and_differentiable() -> None:
    model = subject.build_model()
    assert subject.parameter_count(model) == 5_953
    assert subject.parameter_count(model) < subject.PROTOCOL["maximum_parameter_count"]
    probabilities = torch.zeros((1, 9, 3), requires_grad=True)
    with torch.no_grad():
        probabilities[:, :, 0] = 1
        probabilities[:, 0] = torch.tensor([0.0, 1.0, 0.0])
        probabilities[:, 1] = torch.tensor([0.0, 0.0, 1.0])
    joint = subject.joint_count_probabilities(probabilities)
    assert float(joint[0, 1, 1].detach()) == 1.0
    assert float(joint.sum().detach()) == 1.0
    assert torch.count_nonzero(joint[0][
        torch.tensor([[straight + arc > 9 for arc in range(10)] for straight in range(10)])
    ]) == 0
    joint[0, 1, 1].backward()
    assert probabilities.grad is not None


def test_authority_intervals_and_multiplicities_are_consumed_without_pixels() -> None:
    targets = subject.authority_panel_targets(_authority_panel())
    assert [target["kind"] for target in targets] == ["line", "line", "arc"]
    assert targets[0]["line_length"] == {
        "center": 0.4,
        "lower": 0.3995,
        "upper": 0.4005,
    }
    assert targets[2]["arc_sweep_magnitude"]["center"] == 0.5
    assert {target["shape_membership_local_index"] for target in targets} == {0, 1}
    with pytest.raises(subject.TinyLocalObserverError, match="authority_GAP"):
        subject.authority_panel_targets(_authority_panel(disposition="GAP"))


def test_interval_loss_is_zero_inside_and_gap_never_becomes_none() -> None:
    prediction = torch.tensor(0.5)
    assert float(subject._distance_outside_interval(prediction, _interval(0.5))) == 0.0
    assert float(subject._distance_outside_interval(torch.tensor(0.7), _interval(0.5))) > 0
    certified = _authority_panel(panel_id="hd/task/0/0.png")
    gap = _authority_panel(panel_id="hd/task/0/1.png", disposition="GAP")
    coverage = subject.audit_supervision_coverage((certified, gap))
    assert coverage["eligible_panel_counts"] == {"train": 1, "validation": 0}
    assert coverage["gap_code_counts"] == {"unsupported_style": 1}
    assert coverage["gap_rows_coerced_to_none_or_zero"] == 0


def test_set_loss_backpropagates_through_slots_counts_and_catalog() -> None:
    model = subject.build_model().train()
    pixels = torch.rand((2, 2, 64, 64), generator=torch.Generator().manual_seed(3))
    output = model(pixels)
    assert output["attention"].shape == (2, 9, 8, 8)
    assert output["geometry"].shape == (2, 9, 3)
    losses = subject.set_prediction_loss(
        output, [_targets(), _targets()], torch.tensor([1, 2], dtype=torch.long)
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.queries.grad is not None


def test_raw_evidence_binds_exact_png_size_and_state() -> None:
    model = subject.build_model().eval()
    state_digest = subject.state_dict_digest(model.state_dict())
    raw_png = _png()
    records = subject.predict_raw_evidence(
        model,
        panel_ids=["hd/task/0/0.png"],
        panel_png_bytes=[raw_png],
        checkpoint_state_dict_sha256=state_digest,
        config_digest=subject.successor_config_digest(),
    )
    assert records[0]["png_size_bytes"] == len(raw_png)
    assert records[0]["png_sha256"] == subject._address(raw_png)
    assert len(records[0]["slots"]) == 9
    with pytest.raises(subject.TinyLocalObserverError, match="model state"):
        subject.predict_raw_evidence(
            model,
            panel_ids=["hd/task/0/0.png"],
            panel_png_bytes=[raw_png],
            checkpoint_state_dict_sha256=SHA_A,
            config_digest=subject.successor_config_digest(),
        )


def test_checkpoint_without_training_authority_is_rejected(tmp_path) -> None:
    model = subject.build_model()
    payload = {
        "architecture_id": subject.ARCHITECTURE_ID,
        "config_digest": subject.successor_config_digest(),
        "selected_epoch": 0,
        "source_sha256": subject.source_sha256(),
        "state_dict": model.state_dict(),
        "state_dict_sha256": subject.state_dict_digest(model.state_dict()),
        "training_precommit_record_digest": SHA_A,
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    with pytest.raises(subject.TinyLocalObserverError, match="authority is absent"):
        subject.load_verified_checkpoint(
            path,
            expected_training_precommit_record_digest=None,
            training_result=None,
            expected_training_result_record_digest=None,
        )
    result = subject._seal(
        {
            "checkpoint_raw_sha256": subject._address(path.read_bytes()),
            "checkpoint_state_dict_sha256": payload["state_dict_sha256"],
            "config_digest": payload["config_digest"],
            "schema": "gkm.bongard-tiny-local-action-development-result.v1",
            "selected_epoch": 0,
            "training_precommit_record_digest": SHA_A,
            "validation_gate": {"passed": True},
        }
    )
    loaded, envelope, _digest = subject.load_verified_checkpoint(
        path,
        expected_training_precommit_record_digest=SHA_A,
        training_result=result,
        expected_training_result_record_digest=result["record_digest"],
    )
    assert subject.state_dict_digest(loaded.state_dict()) == envelope["state_dict_sha256"]
    failed_body = dict(result)
    failed_body.pop("record_digest")
    failed_body["validation_gate"] = {"passed": False}
    failed = subject._seal(failed_body)
    with pytest.raises(subject.TinyLocalObserverError, match="gate did not pass"):
        subject.load_verified_checkpoint(
            path,
            expected_training_precommit_record_digest=SHA_A,
            training_result=failed,
            expected_training_result_record_digest=failed["record_digest"],
        )
    diagnostic, _envelope, _digest = subject.load_verified_checkpoint(
        path,
        expected_training_precommit_record_digest=SHA_A,
        training_result=failed,
        expected_training_result_record_digest=failed["record_digest"],
        require_passed_development_gate=False,
    )
    assert subject.state_dict_digest(diagnostic.state_dict()) == payload["state_dict_sha256"]


def test_joint_calibration_and_cold_replay_are_model_free() -> None:
    raw = [
        _raw_record(panel_id=f"hd/task/0/{index}.png", line=2, arc=1, catalog=2)
        for index in range(3)
    ]
    truth = [
        {
            "arc_action_count": 1,
            "catalog_class_index": 2,
            "panel_id": value["panel_id"],
            "straight_action_count": 2,
        }
        for value in raw
    ]
    calibration = subject.fit_joint_calibrator(
        raw, truth, alpha=0.5, calibration_manifest_record_digest=SHA_A
    )
    calibrated = subject.apply_joint_calibrator(raw[0], calibration)
    assert calibrated["joint_straight_arc_catalog_candidates"] == [[2, 1, 2]]
    replay = subject.cold_replay_observation(raw[0], calibration, calibrated)
    assert replay["model_calls"] == replay["pixel_reads"] == 0
    changed = dict(raw[0])
    changed_body = dict(changed)
    changed_body.pop("record_digest")
    changed_body["joint_count_probabilities_straight_rows_arc_columns"][9][9] = 0.1
    changed = subject._seal(changed_body)
    with pytest.raises(subject.TinyLocalObserverError, match="impossible count"):
        subject.apply_joint_calibrator(changed, calibration)


def test_runtime_work_bound_refuses_another_long_cpu_run() -> None:
    value = subject.runtime_work_bound(
        training_occurrences=11_200,
        validation_occurrences=1_392,
        measured_seconds_per_frozen_batch=0.1,
    )
    assert value["optimizer_steps"] == 528
    assert value["projected_runtime_seconds_with_3x_margin"] < 420
    with pytest.raises(subject.RuntimeBudgetExceeded, match="seven minutes"):
        subject.runtime_work_bound(
            training_occurrences=11_200,
            validation_occurrences=1_392,
            measured_seconds_per_frozen_batch=1.0,
        )


def test_precommit_is_zero_pixel_and_infrastructure_only(tmp_path, monkeypatch) -> None:
    records = {
        "baseline": {"record_digest": subject.FAILED_BASELINE_DIGEST},
        "spatial": {"record_digest": subject.RETIRED_SPATIAL_OUTCOME_DIGEST},
        "fit": {
            "duplicate_digest_audit": {"effective_training_panel_count": 11_200},
            "effective_validation_panel_count": 1_392,
            "record_digest": (
                "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
            ),
            "validation_removed_due_exact_train_duplicate": {"panel_count": 8},
        },
    }
    original_load = subject._load_record

    def load(path, **kwargs):
        return records[path.name] if path.name in records else original_load(path, **kwargs)

    monkeypatch.setattr(subject, "_load_record", load)
    authority_body = {
        "schema": "gkm.bongard-pose-free-local-action-authority.v1",
        "semantics": {},
    }
    authority = subject._seal(authority_body)
    coverage = {
        "authority_record_digest": authority["record_digest"],
        "capacity_gap_panel_count": 0,
        "cohort_panel_counts": {"train": 11_200, "validation": 1_400},
        "eligible_panel_counts": {"train": 11_200, "validation": 1_400},
        "gap_code_counts": {},
        "gap_rows_coerced_to_none_or_zero": 0,
        "panel_count": 12_600,
        "pixel_aligned_targets_created": 0,
        "scalar_midpoints_substituted_for_intervals": 0,
    }
    conflict = {
        "authority_gap_occurrences": 0,
        "count_and_catalog_supervision_occurrences": 12_592,
        "descriptor_conflict_occurrences": 2,
        "descriptor_eligible_occurrences": 12_590,
        "descriptor_gap_is_never_none_or_zero": True,
        "effective_occurrence_count": 12_592,
    }
    output = tmp_path / "precommit.json"
    value = subject.create_successor_precommit(
        failed_baseline_path=tmp_path / "baseline",
        retired_spatial_outcome_path=tmp_path / "spatial",
        fit_precommit_path=tmp_path / "fit",
        supervision_authority_record=authority,
        supervision_coverage=coverage,
        descriptor_conflict_audit=conflict,
        runtime_probe={
            "frozen_batch_size": 128,
            "median_seconds_per_frozen_batch": 0.1,
            "parameter_count": subject.parameter_count(),
            "synthetic_only": True,
        },
        trainer_source_sha256="a" * 64,
        training_entrypoint_status="live_runnable_development_only",
        intended_checkpoint=tmp_path / "model.pt",
        intended_result=tmp_path / "result.json",
        output=output,
    )
    assert value["pixels_read_by_precommit"] == 0
    assert value["training_entrypoint_status"] == "live_runnable_development_only"
    assert value["runtime_work_bound"]["maximum_wall_runtime_seconds"] == 600.0
