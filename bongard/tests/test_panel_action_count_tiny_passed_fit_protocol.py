"""Focused promotion-boundary tests for the tiny observer fit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_tiny_local_dev_command as core
from bongard import panel_action_count_tiny_local_train_command as trainer
from bongard.panel_action_count_tiny_passed_fit_protocol import (
    TinyPassedFitGap,
    TinyPassedFitProtocol,
    TinyPassedFitProtocolError,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_FAILED_FIT_ROOT = (
    REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/panel_action_count_tiny_local_20260810_v1"
)


def _seal(body):
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _metrics(*, passed: bool) -> dict[str, object]:
    return {
        "arc_top1": 0.90,
        "catalog_all_class_top1": 0.80,
        "known_catalog_binary_balanced_accuracy": 0.80,
        "panel_occurrences": 1_392,
        "descriptor_deployment_authority": False,
        "descriptor_eligible_digest_groups": 1_392,
        "descriptor_geometry_interval_hit": 0.5,
        "descriptor_geometry_interval_hit_denominator": 10,
        "descriptor_geometry_interval_hit_numerator": 5,
        "descriptor_matched_primitive_accuracy": 0.5,
        "descriptor_matched_primitive_denominator": 10,
        "descriptor_matched_primitive_numerator": 5,
        "descriptor_primitive_multiset_exact": 0.5,
        "descriptor_primitive_multiset_exact_denominator": 1_392,
        "descriptor_primitive_multiset_exact_numerator": 696,
        "straight_top1": 0.80 if passed else 0.60,
    }


def _chain(tmp_path: Path, *, passed: bool = True):
    checkpoint_path = tmp_path / "model.pt"
    precommit_path = tmp_path / "training_precommit.json"
    result_path = tmp_path / "result.json"
    conflict = {
        "all_effective_png_groups_descriptor_loss_eligible": True,
        "authority_gap_occurrences": 0,
        "committed_audit_record_digest": trainer.COMMITTED_CONFLICT_AUDIT_DIGEST,
        "committed_audit_source_sha256": SHA_A,
        "count_and_catalog_supervision_occurrences": 12_592,
        "descriptor_conflict_occurrences": 0,
        "descriptor_eligible_group_counts": {"train": 11_143, "validation": 1_392},
        "descriptor_eligible_occurrences": 12_592,
        "descriptor_gap_is_never_none_or_zero": True,
        "effective_occurrence_count": 12_592,
        "eligibility_index_digests": {"train": SHA_B, "validation": SHA_C},
    }
    intended = {
        "checkpoint": str(checkpoint_path.resolve()),
        "core_precommit": str((tmp_path / "core_precommit.json").resolve()),
        "precommit": str(precommit_path.resolve()),
        "replay": str((tmp_path / "replay.json").resolve()),
        "result": str(result_path.resolve()),
    }
    precommit = _seal(
        {
            "architecture_id": core.ARCHITECTURE_ID,
            "authorization_record_digest": SHA_A,
            "config_digest": core.successor_config_digest(),
            "core_precommit_record_digest": SHA_B,
            "core_source_sha256": core.source_sha256(),
            "decontaminated_occurrence_counts": {"train": 11_200, "validation": 1_392},
            "descriptor_conflict_audit_record_digest": trainer.COMMITTED_CONFLICT_AUDIT_DIGEST,
            "descriptor_target_conflict_audit": conflict,
            "fit_precommit_record_digest": trainer.FIT_PRECOMMIT_DIGEST,
            "forbidden_cohorts": list(core.PROTOCOL["forbidden_cohorts"]),
            "intended_outputs": intended,
            "pixels_read_by_precommit": 0,
            "protocol": json.loads(canonical_json(dict(core.PROTOCOL))),
            "schema": trainer.PRECOMMIT_SCHEMA,
            "source_sha256": trainer.source_sha256(),
            "supervision_authority_record_digest": SHA_C,
        }
    )
    _write(precommit_path, precommit)

    model = core.build_model()
    state_digest = core.state_dict_digest(model.state_dict())
    torch.save(
        {
            "architecture_id": core.ARCHITECTURE_ID,
            "config_digest": core.successor_config_digest(),
            "selected_epoch": 0,
            "source_sha256": core.source_sha256(),
            "state_dict": model.state_dict(),
            "state_dict_sha256": state_digest,
            "training_precommit_record_digest": precommit["record_digest"],
        },
        checkpoint_path,
    )
    checkpoint_digest = core._address(checkpoint_path.read_bytes())
    metrics = _metrics(passed=passed)
    history = [
        {**metrics, "epoch": epoch, "training_group_mean_loss": 1.0 + epoch}
        for epoch in range(int(core.PROTOCOL["epochs"]))
    ]
    result = _seal(
        {
            "architecture_id": core.ARCHITECTURE_ID,
            "authorization_record_digest": precommit["authorization_record_digest"],
            "checkpoint_raw_sha256": checkpoint_digest,
            "checkpoint_state_dict_sha256": state_digest,
            "config_digest": core.successor_config_digest(),
            "decontaminated_occurrence_counts": {"train": 11_200, "validation": 1_392},
            "descriptor_target_conflict_audit": conflict,
            "forbidden_cohorts_opened": 0,
            "history": history,
            "pixel_occurrences_reread": 12_592,
            "runtime_budget": {
                "cooperative_batch_boundary_deadline": True,
                "finalization_reserve_seconds": trainer.FINALIZATION_RESERVE_SECONDS,
                "limit_seconds": float(core.PROTOCOL["maximum_wall_runtime_seconds"]),
                "passed_before_result_seal": True,
            },
            "runtime_seconds": 12.5,
            "schema": trainer.RESULT_SCHEMA,
            "selected_epoch": 0,
            "source_sha256": trainer.source_sha256(),
            "training_precommit_record_digest": precommit["record_digest"],
            "validation_gate": trainer._validation_gate(metrics),
            "validation_metrics": metrics,
            "validation_prediction_rows_digest": SHA_A,
        }
    )
    _write(result_path, result)
    return precommit_path, result_path, checkpoint_path, precommit, result


def _load(path: Path):
    return json.loads(path.read_bytes())


def test_passed_chain_creates_and_fresh_verifies_exact_protocol(tmp_path: Path) -> None:
    precommit_path, result_path, checkpoint_path, _precommit, _result = _chain(tmp_path)
    outcome = TinyPassedFitProtocol.from_files(
        training_precommit_path=precommit_path,
        training_result_path=result_path,
        checkpoint_path=checkpoint_path,
    )
    assert type(outcome) is TinyPassedFitProtocol
    assert outcome.to_data()["development_fit_passed"] is True
    assert outcome.to_data()["calibration_authorized"] is False
    assert outcome.to_data()["benchmark_sealable"] is False
    assert TinyPassedFitProtocol.from_data(outcome.to_data()) == outcome
    assert outcome.verify(
        training_precommit_path=precommit_path,
        training_result_path=result_path,
        checkpoint_path=checkpoint_path,
        expected_protocol_address=outcome.record_digest,
    ) == outcome


def test_exact_failed_result_becomes_gap_and_never_protocol(tmp_path: Path) -> None:
    precommit_path, result_path, checkpoint_path, precommit, result = _chain(
        tmp_path, passed=False
    )
    outcome = TinyPassedFitProtocol.from_files(
        training_precommit_path=precommit_path,
        training_result_path=result_path,
        checkpoint_path=checkpoint_path,
    )
    assert type(outcome) is TinyPassedFitGap
    assert outcome.failed_checks == ("straight_top1",)
    assert outcome.to_data()["development_fit_passed"] is False
    assert outcome.to_data()["support_query_inference_authorized"] is False
    assert TinyPassedFitGap.from_data(outcome.to_data()) == outcome
    assert outcome.verify(
        training_precommit_path=precommit_path,
        training_result_path=result_path,
        checkpoint_path=checkpoint_path,
        expected_gap_address=outcome.record_digest,
    ) == outcome
    with pytest.raises(core.TinyLocalObserverError, match="gate did not pass"):
        core.load_verified_checkpoint(
            checkpoint_path,
            expected_training_precommit_record_digest=precommit["record_digest"],
            training_result=result,
            expected_training_result_record_digest=result["record_digest"],
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("precommit", "config_digest", SHA_A, "precommit policy"),
        ("precommit", "pixels_read_by_precommit", False, "precommit policy"),
        ("result", "training_precommit_record_digest", SHA_B, "result policy"),
        ("result", "forbidden_cohorts_opened", False, "result policy"),
        ("result", "checkpoint_state_dict_sha256", SHA_C, "checkpoint verification"),
    ),
)
def test_resealed_lineage_tampering_is_rejected(
    tmp_path: Path, target: str, field: str, value: object, message: str
) -> None:
    precommit_path, result_path, checkpoint_path, _precommit, _result = _chain(tmp_path)
    path = precommit_path if target == "precommit" else result_path
    changed = _load(path)
    changed.pop("record_digest")
    changed[field] = value
    _write(path, _seal(changed))
    with pytest.raises(TinyPassedFitProtocolError, match=message):
        TinyPassedFitProtocol.from_files(
            training_precommit_path=precommit_path,
            training_result_path=result_path,
            checkpoint_path=checkpoint_path,
        )


def test_claimed_pass_cannot_disagree_with_any_named_check(tmp_path: Path) -> None:
    precommit_path, result_path, checkpoint_path, _precommit, _result = _chain(tmp_path)
    changed = _load(result_path)
    changed.pop("record_digest")
    changed["validation_gate"]["checks"]["straight_top1"] = False
    changed["validation_gate"]["passed"] = True
    _write(result_path, _seal(changed))
    with pytest.raises(TinyPassedFitProtocolError, match="validation gate differs"):
        TinyPassedFitProtocol.from_files(
            training_precommit_path=precommit_path,
            training_result_path=result_path,
            checkpoint_path=checkpoint_path,
        )


def test_checkpoint_bytes_and_json_canonical_bytes_are_reverified(tmp_path: Path) -> None:
    precommit_path, result_path, checkpoint_path, _precommit, _result = _chain(tmp_path)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
    with pytest.raises(TinyPassedFitProtocolError, match="checkpoint verification"):
        TinyPassedFitProtocol.from_files(
            training_precommit_path=precommit_path,
            training_result_path=result_path,
            checkpoint_path=checkpoint_path,
        )

    precommit_path, result_path, checkpoint_path, _precommit, _result = _chain(
        tmp_path / "json"
    )
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(TinyPassedFitProtocolError, match="canonical JSON"):
        TinyPassedFitProtocol.from_files(
            training_precommit_path=precommit_path,
            training_result_path=result_path,
            checkpoint_path=checkpoint_path,
        )


@pytest.mark.skipif(
    not all(
        (LIVE_FAILED_FIT_ROOT / name).exists()
        for name in ("training_precommit.json", "result.json", "model.pt")
    ),
    reason="exact live tiny failed-fit artifacts are absent",
)
def test_exact_live_failed_fit_is_bound_as_gap_not_protocol() -> None:
    outcome = TinyPassedFitProtocol.from_files(
        training_precommit_path=LIVE_FAILED_FIT_ROOT / "training_precommit.json",
        training_result_path=LIVE_FAILED_FIT_ROOT / "result.json",
        checkpoint_path=LIVE_FAILED_FIT_ROOT / "model.pt",
    )
    assert type(outcome) is TinyPassedFitGap
    assert outcome.training_precommit_record_digest == (
        "sha256:f23f7217b23614d74cf25a972546160fbb9635a94808de3ffa594e188e56160d"
    )
    assert outcome.training_precommit_file_sha256 == (
        "sha256:a0a191ded67e827f63a42d231cec30150460ac21f4ac1bc3eb2be9cfe48aa137"
    )
    assert outcome.training_result_record_digest == (
        "sha256:48e0e6404ba0f070712abad25e0219ceea7482d396782a21d07b7e898734b824"
    )
    assert outcome.training_result_file_sha256 == (
        "sha256:8978897cdad9e89612500e17ad6ee88b8a963765c1665cf0abae2a7628effafa"
    )
    assert outcome.checkpoint_raw_sha256 == (
        "sha256:6f8934122e25b271a8539388e2e47413905c5536e2b0c0c6af3f46cf6bd3c8d5"
    )
    assert outcome.failed_checks == (
        "arc_top1",
        "known_catalog_binary_balanced_accuracy",
        "straight_top1",
    )
    assert outcome.record_digest == (
        "sha256:7dca055143b3a687f477bf94f4b8763a4403292269a59887579b686e08a566b1"
    )
