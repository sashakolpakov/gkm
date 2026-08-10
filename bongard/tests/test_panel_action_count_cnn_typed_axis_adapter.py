from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_action_count_cnn_postprediction_labels_v3 import contract_digest
from bongard.panel_action_count_cnn_typed_axis_adapter import (
    ARCHITECTURE_ID,
    CNNObserverProtocol,
    CNNPopulationGrant,
    CNNTypedAxisMatrixArtifact,
    CNNToTypedAxisError,
    FINAL_TRAINER_SOURCE_SHA256,
    FrozenSupportPredictionBatch,
    GENERIC_V3_PLAN_RECORD_DIGEST,
    GENERIC_V3_PLAN_SOURCE_SHA256,
    PopulationScope,
    SAME_FAMILY_CALIBRATION_TASK_IDS,
    SAME_FAMILY_PREREG_RECORD_DIGEST,
    SAME_FAMILY_PREREG_SOURCE_SHA256,
    SUPPORT_ORDINALS,
    SupportPanelPrediction,
    TARGET_TASK_ID,
    adapter_algorithm_record,
    build_cnn_typed_support_matrix,
    cold_replay_cnn_typed_support_matrix,
    observer_protocol_from_fit_artifacts,
)
from bongard.panel_typed_axis_slate_v2 import (
    MAX_FORMULA_COUNT,
    Axis,
    EvidenceKind,
    SupportSide,
    TypedAxisInventory,
)


def _a(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _sealed(body: dict[str, object]) -> dict[str, object]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _fit_artifacts(
    *, passed: bool = True
) -> tuple[dict[str, object], str, dict[str, object], str]:
    removed = [{"validation_panel_id": f"panel-{index}"} for index in range(8)]
    precommit_body: dict[str, object] = {
        "schema": "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v2",
        "trainer_source_sha256": FINAL_TRAINER_SOURCE_SHA256,
        "validation_decontamination_gate": {"passed": passed},
        "effective_training_panel_count": 11_200,
        "effective_validation_panel_count": 1_392,
        "validation_removed_due_exact_train_duplicate": removed,
    }
    precommit = _sealed(precommit_body)
    precommit_raw = canonical_json(precommit) + b"\n"
    precommit_source = "sha256:" + hashlib.sha256(precommit_raw).hexdigest()
    gate = {
        "checks": {
            "arc_top1": passed,
            "known_catalog_binary_balanced_accuracy": passed,
            "straight_top1": passed,
        },
        "passed": passed,
    }
    body: dict[str, object] = {
        "schema": "gkm.bongard-action-count-catalog-cnn-fit-result.v2",
        "architecture_id": ARCHITECTURE_ID,
        "checkpoint_raw_sha256": _a("checkpoint-raw"),
        "checkpoint_state_dict_sha256": _a("checkpoint-state"),
        "config_digest": _a("config"),
        "validation_gate": gate,
        "fit_precommit_record_digest": precommit["record_digest"],
        "adaptive_post_exposure_development_correction": {
            "validation_decontamination_gate": {"passed": passed},
            "effective_training_panel_count": 11_200,
            "effective_validation_panel_count": 1_392,
            "validation_removed_due_exact_train_duplicate": removed,
        },
    }
    record = _sealed(body)
    raw = canonical_json(record) + b"\n"
    return (
        precommit,
        precommit_source,
        record,
        "sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _protocol() -> CNNObserverProtocol:
    precommit, precommit_source, result, source = _fit_artifacts()
    return observer_protocol_from_fit_artifacts(
        fit_precommit=precommit,
        fit_precommit_source_sha256=precommit_source,
        fit_result=result,
        fit_result_source_sha256=source,
        inference_source_sha256=_a("future-passing-inference-runner-source"),
        postprediction_contract_digest=contract_digest(),
    )


def _grant(
    protocol: CNNObserverProtocol,
    *,
    scope: PopulationScope = PopulationScope.SAME_FAMILY_CONVEX_FOUR_LINES,
    authorized_task_ids: tuple[str, ...] | None = None,
    q: float = 0.6,
) -> CNNPopulationGrant:
    external = _a(f"external-grant-{scope.value}")
    if scope is PopulationScope.SAME_FAMILY_CONVEX_FOUR_LINES:
        calibration_ids = SAME_FAMILY_CALIBRATION_TASK_IDS
        prereg_record = SAME_FAMILY_PREREG_RECORD_DIGEST
        prereg_source = SAME_FAMILY_PREREG_SOURCE_SHA256
        allowed = (TARGET_TASK_ID,) if authorized_task_ids is None else authorized_task_ids
        target_authorization = _a("target-release")
    else:
        calibration_ids = tuple(
            f"hd_generic-calibration-{index:03d}_0000" for index in range(100)
        )
        prereg_record = GENERIC_V3_PLAN_RECORD_DIGEST
        prereg_source = GENERIC_V3_PLAN_SOURCE_SHA256
        allowed = (
            ("hd_generic-support_9000",)
            if authorized_task_ids is None
            else authorized_task_ids
        )
        target_authorization = None
    prediction_record = _a(f"cal-prediction-record-{scope.value}")
    prediction_source = _a(f"cal-prediction-source-{scope.value}")
    return CNNPopulationGrant(
        scope=scope,
        protocol_address=protocol.protocol_address,
        external_grant_record_digest=external,
        external_grant_source_sha256=_a(f"external-grant-source-{scope.value}"),
        calibration_prediction_record_digest=prediction_record,
        calibration_prediction_source_sha256=prediction_source,
        calibration_label_record_digest=_a(f"cal-label-record-{scope.value}"),
        calibration_label_source_sha256=_a(f"cal-label-source-{scope.value}"),
        label_bound_prediction_record_digest=prediction_record,
        label_bound_prediction_source_sha256=prediction_source,
        deployment_joint_q=q,
        calibration_task_ids=calibration_ids,
        scope_preregistration_record_digest=prereg_record,
        scope_preregistration_source_sha256=prereg_source,
        population_release_record_digest=_a(f"release-record-{scope.value}"),
        population_release_source_sha256=_a(f"release-source-{scope.value}"),
        population_release_grant_record_digest=external,
        authorized_task_ids=allowed,
        target_release_authorization_address=target_authorization,
    )


def _peaked(size: int, index: int, mass: float = 0.82) -> tuple[float, ...]:
    remainder = (1.0 - mass) / (size - 1)
    return tuple(mass if item == index else remainder for item in range(size))


def _catalog(mode: str) -> tuple[float, ...]:
    return {
        "convex": (0.09, 0.09, 0.82),
        "nonconvex": (0.09, 0.82, 0.09),
        "both": (0.10, 0.45, 0.45),
        "unresolved": (0.45, 0.45, 0.10),
        "empty": (1.0 / 3.0,) * 3,
    }[mode]


def _class_set(probabilities: tuple[float, ...], q: float) -> tuple[int, ...]:
    return tuple(
        index for index, probability in enumerate(probabilities)
        if 1.0 - probability <= q
    )


def _row(
    task_id: str,
    side: SupportSide,
    ordinal: int,
    *,
    straight: int = 4,
    catalog: str = "convex",
    q: float = 0.6,
) -> SupportPanelPrediction:
    straight_probabilities = _peaked(10, straight)
    catalog_probabilities = _catalog(catalog)
    folder = 1 if side is SupportSide.PRIMARY else 0
    panel_id = f"hd/{task_id}/{folder}/{ordinal}.png"
    return SupportPanelPrediction(
        panel_id=panel_id,
        side=side,
        ordinal=ordinal,
        png_sha256=_a(f"png:{panel_id}"),
        png_size_bytes=1000 + folder * 100 + ordinal,
        straight_logits=tuple(math.log(value) for value in straight_probabilities),
        straight_probabilities=straight_probabilities,
        straight_class_set=_class_set(straight_probabilities, q),
        catalog_logits=tuple(math.log(value) for value in catalog_probabilities),
        catalog_probabilities=catalog_probabilities,
        catalog_class_set=_class_set(catalog_probabilities, q),
    )


def _rows(task_id: str, *, q: float = 0.6) -> tuple[SupportPanelPrediction, ...]:
    primary = tuple(
        _row(task_id, SupportSide.PRIMARY, ordinal, q=q)
        for ordinal in SUPPORT_ORDINALS
    )
    contrast = tuple(
        _row(
            task_id,
            SupportSide.CONTRAST,
            ordinal,
            straight=3 if index < 3 else 4,
            catalog="convex" if index < 3 else "nonconvex",
            q=q,
        )
        for index, ordinal in enumerate(SUPPORT_ORDINALS)
    )
    return primary + contrast


def _batch(
    protocol: CNNObserverProtocol,
    grant: CNNPopulationGrant,
    *,
    task_id: str = TARGET_TASK_ID,
    rows: tuple[SupportPanelPrediction, ...] | None = None,
) -> FrozenSupportPredictionBatch:
    target_authorization = (
        grant.target_release_authorization_address
        if grant.target_release_authorization_address is not None
        else _a(f"generic-task-authorization:{task_id}")
    )
    assert target_authorization is not None
    return FrozenSupportPredictionBatch(
        task_id=task_id,
        protocol_address=protocol.protocol_address,
        population_grant_address=grant.grant_address,
        external_grant_record_digest=grant.external_grant_record_digest,
        prediction_record_digest=_a(f"prediction-record:{task_id}"),
        prediction_source_sha256=_a(f"prediction-source:{task_id}"),
        pixel_precommit_record_digest=_a(f"precommit-record:{task_id}"),
        pixel_precommit_source_sha256=_a(f"precommit-source:{task_id}"),
        target_authorization_record_digest=target_authorization,
        target_authorization_source_sha256=_a(f"authorization-source:{task_id}"),
        checkpoint_state_dict_sha256=protocol.checkpoint_state_dict_sha256,
        config_digest=protocol.config_digest,
        joint_q=grant.deployment_joint_q,
        rows=_rows(task_id, q=grant.deployment_joint_q) if rows is None else rows,
    )


def test_failed_fit_cannot_create_observer_protocol_without_any_cal_output() -> None:
    precommit, precommit_source, failed, source = _fit_artifacts(passed=False)
    with pytest.raises(CNNToTypedAxisError, match="did not pass"):
        observer_protocol_from_fit_artifacts(
            fit_precommit=precommit,
            fit_precommit_source_sha256=precommit_source,
            fit_result=failed,
            fit_result_source_sha256=source,
            inference_source_sha256=_a("runner"),
            postprediction_contract_digest=contract_digest(),
        )

    passed = _protocol()
    with pytest.raises(CNNToTypedAxisError, match="failed fit"):
        replace(passed, fit_validation_gate_passed=False)


def test_generic_v3_scope_cannot_whitelist_or_adapt_target_family() -> None:
    protocol = _protocol()
    with pytest.raises(CNNToTypedAxisError, match="generic fresh-V3 population"):
        _grant(
            protocol,
            scope=PopulationScope.GENERIC_FRESH_V3,
            authorized_task_ids=(TARGET_TASK_ID,),
        )

    generic = _grant(protocol, scope=PopulationScope.GENERIC_FRESH_V3)
    target_batch = _batch(protocol, generic, task_id=TARGET_TASK_ID)
    with pytest.raises(CNNToTypedAxisError, match="generic fresh-V3 grant"):
        build_cnn_typed_support_matrix(
            protocol=protocol,
            population_grant=generic,
            prediction_batch=target_batch,
        )


def test_generic_scope_can_only_adapt_its_exact_non_target_task() -> None:
    protocol = _protocol()
    generic = _grant(protocol, scope=PopulationScope.GENERIC_FRESH_V3)
    task_id = "hd_generic-support_9000"
    batch = _batch(protocol, generic, task_id=task_id)
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol, population_grant=generic, prediction_batch=batch
    )
    assert len(artifact.matrix.rows) == 12
    assert all(TARGET_TASK_ID not in row.row_key for row in artifact.matrix.rows)


def test_same_family_grant_is_only_target_calibrated_set_path() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol,
        population_grant=grant,
        prediction_batch=_batch(protocol, grant),
    )
    for row in artifact.matrix.rows:
        straight = row.cell(Axis.STRAIGHT_ACTION_COUNT)
        catalog = row.cell(Axis.CATALOG_CONVEXITY)
        assert straight.evidence_kind is EvidenceKind.CALIBRATED_SET
        assert catalog.evidence_kind is EvidenceKind.CALIBRATED_SET
        assert straight.calibration_grant_address == grant.grant_address
        assert catalog.calibration_grant_address == grant.grant_address
        for axis in set(Axis) - {Axis.STRAIGHT_ACTION_COUNT, Axis.CATALOG_CONVEXITY}:
            cell = row.cell(axis)
            assert cell.evidence_kind is EvidenceKind.GAP
            assert cell.gap_reason_code == "cnn_axis_not_observed"
            assert cell.observer_protocol_digest == straight.observer_protocol_digest


def test_catalog_projection_handles_empty_unresolved_pair_and_singleton() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    rows = list(_rows(TARGET_TASK_ID))
    modes = ("empty", "unresolved", "both", "nonconvex")
    for index, mode in enumerate(modes):
        original = rows[index]
        probabilities = _catalog(mode)
        rows[index] = replace(
            original,
            catalog_logits=tuple(math.log(value) for value in probabilities),
            catalog_probabilities=probabilities,
            catalog_class_set=_class_set(probabilities, grant.deployment_joint_q),
        )
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol,
        population_grant=grant,
        prediction_batch=_batch(protocol, grant, rows=tuple(rows)),
    )
    cells = [
        artifact.matrix.rows[index].cell(Axis.CATALOG_CONVEXITY)
        for index in range(4)
    ]
    assert cells[0].evidence_kind is EvidenceKind.ERROR
    assert cells[0].error_code == "empty_catalog_class_set"
    assert cells[1].evidence_kind is EvidenceKind.GAP
    assert cells[1].gap_reason_code == "catalog_set_contains_unresolved"
    assert cells[2].possible_values == ("catalog_nonconvex", "catalog_convex")
    assert cells[3].possible_values == ("catalog_nonconvex",)


def test_empty_straight_set_is_error_not_absence() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    rows = list(_rows(TARGET_TASK_ID))
    probabilities = (0.1,) * 10
    rows[0] = replace(
        rows[0],
        straight_logits=tuple(math.log(value) for value in probabilities),
        straight_probabilities=probabilities,
        straight_class_set=(),
    )
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol,
        population_grant=grant,
        prediction_batch=_batch(protocol, grant, rows=tuple(rows)),
    )
    cell = artifact.matrix.rows[0].cell(Axis.STRAIGHT_ACTION_COUNT)
    assert cell.evidence_kind is EvidenceKind.ERROR
    assert cell.error_code == "empty_straight_class_set"


def test_q_role_and_malformed_panel_byte_binding_are_rejected() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    rows = list(_rows(TARGET_TASK_ID))
    rows[0] = replace(rows[0], straight_class_set=(3,))
    with pytest.raises(CNNToTypedAxisError, match="straight class set differs"):
        _batch(protocol, grant, rows=tuple(rows))

    roles = list(_rows(TARGET_TASK_ID))
    with pytest.raises(CNNToTypedAxisError, match="role does not match"):
        replace(roles[0], side=SupportSide.CONTRAST)

    with pytest.raises(CNNToTypedAxisError, match="support PNG"):
        replace(roles[0], png_sha256="not-an-address")


def test_target_release_authorization_is_bound_to_same_family_grant() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    batch = replace(
        _batch(protocol, grant),
        target_authorization_record_digest=_a("foreign-target-authorization"),
    )
    with pytest.raises(CNNToTypedAxisError, match="custody differs"):
        build_cnn_typed_support_matrix(
            protocol=protocol,
            population_grant=grant,
            prediction_batch=batch,
        )


def test_target_support_derives_full_target_independent_1366_inventory() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol,
        population_grant=grant,
        prediction_batch=_batch(protocol, grant),
    )
    inventory = TypedAxisInventory.derive(artifact.matrix)
    assert len(inventory.formulas) == MAX_FORMULA_COUNT == 1366
    assert inventory.to_data()["query_rows_seen"] == 0
    assert inventory.to_data()["model_calls_for_derivation_or_replay"] == 0
    expected = [
        formula
        for formula in inventory.formulas
        if tuple((atom.axis, atom.value) for atom in formula.atoms)
        == (
            (Axis.STRAIGHT_ACTION_COUNT, 4),
            (Axis.CATALOG_CONVEXITY, "catalog_convex"),
        )
    ]
    assert len(expected) == 1
    assert expected[0].admitted is True
    assert expected[0].formula_id in inventory.admitted_formula_ids
    assert adapter_algorithm_record()["query_rows_seen"] == 0


def test_artifact_cold_replay_is_zero_call_and_tamper_detecting() -> None:
    protocol = _protocol()
    grant = _grant(protocol)
    artifact = build_cnn_typed_support_matrix(
        protocol=protocol,
        population_grant=grant,
        prediction_batch=_batch(protocol, grant),
    )
    restored = cold_replay_cnn_typed_support_matrix(
        artifact, expected_artifact_address=artifact.artifact_address
    )
    assert restored == artifact
    data = artifact.to_data()
    assert data["png_reads_during_adaptation_or_replay"] == 0
    assert data["model_calls_during_adaptation_or_replay"] == 0
    assert data["label_source_calls_during_adaptation_or_replay"] == 0
    assert data["lean_present"] is False

    changed = artifact.to_data()
    changed["prediction_batch"]["rows"][0]["png_size_bytes"] += 1
    with pytest.raises(CNNToTypedAxisError):
        CNNTypedAxisMatrixArtifact.from_data(changed)
    with pytest.raises(CNNToTypedAxisError, match="address differs"):
        cold_replay_cnn_typed_support_matrix(
            artifact, expected_artifact_address=_a("wrong-artifact")
        )
