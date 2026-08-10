"""Focused tests for the development-only spatial action observer."""

from __future__ import annotations

import json

import torch

from bongard import panel_action_count_spatial_dev_command as subject


def test_semantic_channels_are_deterministic_and_coarse() -> None:
    ink = torch.zeros((2, 1, 128, 128), dtype=torch.uint8)
    ink[:, :, 60:63, 15:113] = 255
    ink[1, :, 55:58, 20:25] = 255
    first = subject.semantic_channels(ink)
    second = subject.semantic_channels(ink.clone())
    assert torch.equal(first, second)
    assert tuple(first.shape) == (2, 3, 128, 128)
    binary, coarse = first[:, 1], first[:, 2]
    binary_variation = (binary[:, 1:] - binary[:, :-1]).abs().sum()
    coarse_variation = (coarse[:, 1:] - coarse[:, :-1]).abs().sum()
    assert coarse_variation < binary_variation
    diagnostic = subject.decoration_invariance_diagnostic()
    assert diagnostic["passed"] is True
    assert diagnostic["maximum_observed_coarse_to_raw_ratio"] <= 0.20


def test_count_heads_are_mechanically_isolated_from_raw_and_binary() -> None:
    model = subject.build_model(seed=260810).eval()
    value = torch.rand((2, 3, 128, 128), generator=torch.Generator().manual_seed(9))
    changed = value.clone()
    changed[:, :2] = 1.0 - changed[:, :2]
    with torch.no_grad():
        first = model(value)
        second = model(changed)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[2], second[2])


def test_effective_number_weights_upweight_rare_classes() -> None:
    groups = [
        subject.base.MaterializedGroup(
            cohort="train",
            digest=f"sha256:{index:064x}",
            ink=torch.zeros((128, 128), dtype=torch.uint8).numpy(),
            labels=(label, 0, 0),
            multiplicity=multiplicity,
        )
        for index, (label, multiplicity) in enumerate(((0, 1000), (1, 10)))
    ]
    weights, counts = subject._effective_class_weights(groups, 0, 3)
    assert counts == [1000, 10, 0]
    assert float(weights[1]) > float(weights[0]) > 0
    assert float(weights[2]) == 0


def test_precommit_is_source_bound_and_reads_no_pixels(tmp_path, monkeypatch) -> None:
    authorization = {"record_digest": subject.FIT_AUTHORIZATION_DIGEST}
    precommit = {"record_digest": subject.FIT_PRECOMMIT_DIGEST}
    failed = {"record_digest": subject.FAILED_FIT_DIGEST}
    replay = {"record_digest": subject.FAILED_REPLAY_DIGEST}
    monkeypatch.setattr(
        subject,
        "_verify_predecessor",
        lambda **_: (authorization, precommit, failed, replay),
    )
    output = tmp_path / "precommit.json"
    value = subject.create_successor_precommit(
        fit_authorization=tmp_path / "authorization.json",
        fit_precommit=tmp_path / "fit-precommit.json",
        failed_fit=tmp_path / "fit.json",
        failed_replay=tmp_path / "replay.json",
        intended_checkpoint=tmp_path / "model.pt",
        intended_inner_result=tmp_path / "inner.json",
        intended_result=tmp_path / "result.json",
        output=output,
    )
    assert value["pixels_read_by_precommit"] == 0
    assert value["development_occurrence_counts"] == {
        "train": 11_200,
        "validation": 1_392,
    }
    assert "fresh_v3_calibration" in value["forbidden_cohorts"]
    assert "target" in value["forbidden_cohorts"]
    assert json.loads(output.read_bytes()) == value
    assert subject.create_successor_precommit(
        fit_authorization=tmp_path / "authorization.json",
        fit_precommit=tmp_path / "fit-precommit.json",
        failed_fit=tmp_path / "fit.json",
        failed_replay=tmp_path / "replay.json",
        intended_checkpoint=tmp_path / "model.pt",
        intended_inner_result=tmp_path / "inner.json",
        intended_result=tmp_path / "result.json",
        output=output,
    ) == value


def test_stratum_gate_is_additional_to_failed_or_passed_base_gate() -> None:
    strata = {
        name: {"straight_top1": threshold}
        for name, threshold in subject.STRATUM_THRESHOLDS.items()
    }
    inner = {
        "validation_gate": {"passed": True},
        "validation_metrics": {"straight_required_strata": strata},
    }
    assert subject._successor_gate(inner)["passed"] is True
    inner["validation_metrics"]["straight_required_strata"][
        "straight_true_count_4"
    ]["straight_top1"] = 0.49
    assert subject._successor_gate(inner)["passed"] is False


def test_successor_install_restores_parent_runtime_and_protocol_verifier() -> None:
    original = {
        "architecture": subject.base.ARCHITECTURE_ID,
        "protocol": subject.base.EXECUTION_PROTOCOL,
        "preprocess": subject.base.preprocess_png_bytes,
        "batch": subject.base._batch_tensor,
        "weights": subject.base._class_weights,
        "model": subject.base.build_model,
        "verify": subject.base._verify_execution_protocol,
    }
    with subject._installed_successor():
        assert subject.base.ARCHITECTURE_ID == subject.ARCHITECTURE_ID
        assert subject.base.EXECUTION_PROTOCOL["epochs"] == 24
        assert subject.base._verify_execution_protocol is not original["verify"]
    assert subject.base.ARCHITECTURE_ID == original["architecture"]
    assert subject.base.EXECUTION_PROTOCOL is original["protocol"]
    assert subject.base.preprocess_png_bytes is original["preprocess"]
    assert subject.base._batch_tensor is original["batch"]
    assert subject.base._class_weights is original["weights"]
    assert subject.base.build_model is original["model"]
    assert subject.base._verify_execution_protocol is original["verify"]
