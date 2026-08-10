"""Exact extracted-panel custody for the typed-axis production pipeline.

This module is deliberately an integration boundary, not an observer.  It
joins an already-produced, deterministically replayable typed-axis matrix to
the exact twelve support PNGs released by the authenticated extracted-tree
gate.  The query counterpart joins one post-commit released query PNG to one
frozen prediction row and deterministically projects the same typed cells.

The current closed observation backend is the calibrated CNN adapter.  A new
backend must be added as an exact type branch with its own cold replay; opaque
observer dictionaries and training-only supervision records are rejected.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
)
from bongard.official_extracted_panel_archive import (
    OfficialExtractedPanelArchive,
    ReleasedOfficialExtractedPanel,
)
from bongard.panel_action_count_cnn_typed_axis_adapter import (
    CNNPopulationGrant,
    CNNTypedAxisMatrixArtifact,
    SupportPanelPrediction,
    cold_replay_cnn_typed_support_matrix,
)
from bongard.panel_feature_extracted_release_gate import (
    PanelFeatureExtractedExecutionPrecommit,
    PanelFeatureExtractedReleaseAuthorization,
)
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    SupportSide,
    TypedAxisCell,
    TypedSupportMatrix,
    TypedSupportRow,
)


SUPPORT_CUSTODY_SCHEMA = "gkm.bongard-typed-axis-support-custody.v2"
QUERY_OBSERVATION_SCHEMA = "gkm.bongard-typed-axis-query-observation.v2"
SUPPORT_ORDINALS = (0, 1, 2, 3, 5, 6)
QUERY_ORDINAL = 4

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TypedAxisCustodyV2Error(RuntimeError):
    """A released pixel, observer projection, role, or durable receipt differs."""


def panel_typed_axis_custody_v2_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TypedAxisCustodyV2Error(f"{label} must be a sha256: address")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise TypedAxisCustodyV2Error(f"{label} fields differ")
    return value


def _receipt_binds(
    receipt: ObjectBongardWriteOnceReceipt,
    released: ReleasedOfficialExtractedPanel,
    *,
    object_kind: str,
) -> None:
    if type(receipt) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("released panel custody needs an exact store receipt")
    payload = canonical_json(released.to_data()) + b"\n"
    if (
        receipt.object_kind != object_kind
        or receipt.object_digest != released.record_digest
        or receipt.payload_digest
        != "sha256:" + hashlib.sha256(payload).hexdigest()
        or receipt.size_bytes != len(payload)
    ):
        raise TypedAxisCustodyV2Error("released panel store receipt differs")


def _expected_support_ids(task: ObjectBongardTaskPlan) -> tuple[str, ...]:
    # The frozen batch plan names generator-positive folder ``1`` as side_0
    # and generator-negative folder ``0`` as side_1.  Typed PRIMARY is the
    # former and typed CONTRAST is the latter.  Pin paths, not suggestive side
    # names, so a future naming cleanup cannot silently reverse polarity.
    primary = tuple(
        f"{task.family}/{task.task_id}/1/{ordinal}.png" for ordinal in SUPPORT_ORDINALS
    )
    contrast = tuple(
        f"{task.family}/{task.task_id}/0/{ordinal}.png" for ordinal in SUPPORT_ORDINALS
    )
    expected = primary + contrast
    if (
        task.side_0_query_panel_id
        != f"{task.family}/{task.task_id}/1/{QUERY_ORDINAL}.png"
        or task.side_1_query_panel_id
        != f"{task.family}/{task.task_id}/0/{QUERY_ORDINAL}.png"
        or task.side_0_support_panel_ids != primary
        or task.side_1_support_panel_ids != contrast
    ):
        raise TypedAxisCustodyV2Error(
            "typed-axis v2 task plan folder roles or ordinal-4 queries differ"
        )
    return expected


def _support_content(value: "TaskBoundTypedAxisSupportArtifact") -> dict[str, object]:
    return {
        "schema": SUPPORT_CUSTODY_SCHEMA,
        "source_digest": panel_typed_axis_custody_v2_source_digest(),
        "task_plan": value.task_plan.to_data(),
        "task_plan_digest": value.task_plan.record_digest,
        "execution_precommit": value.execution_precommit.to_data(),
        "execution_precommit_digest": value.execution_precommit.record_digest,
        "release_authorization": value.release_authorization.to_data(),
        "release_authorization_digest": value.release_authorization.record_digest,
        "released_support_panels": [item.to_data() for item in value.released_support_panels],
        "released_support_store_receipts": [item.to_data() for item in value.released_support_store_receipts],
        "observer_matrix_artifact": value.observer_matrix_artifact.to_data(),
        "observer_matrix_artifact_address": value.observer_matrix_artifact.artifact_address,
        "matrix_address": value.matrix.matrix_address,
        "support_role_order": "task-side0-folder1-primary_then-task-side1-folder0-contrast",
        "support_panel_count": 12,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_panel_count": 0,
        "support_pixels_and_observations_bound_before_inventory": True,
        "opaque_observer_artifacts_accepted": False,
        "training_supervision_artifacts_accepted": False,
        "observer_inference_externally_authenticated": False,
        "benchmark_sealable": False,
        "query_release_authorized": False,
        "query_pixels_seen": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
    }


@dataclass(frozen=True, slots=True)
class TaskBoundTypedAxisSupportArtifact:
    """Twelve exact released support PNGs joined to one replayable matrix."""

    task_plan: ObjectBongardTaskPlan
    execution_precommit: PanelFeatureExtractedExecutionPrecommit
    release_authorization: PanelFeatureExtractedReleaseAuthorization
    released_support_panels: tuple[ReleasedOfficialExtractedPanel, ...]
    released_support_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...]
    observer_matrix_artifact: CNNTypedAxisMatrixArtifact
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_plan) is not ObjectBongardTaskPlan:
            raise TypeError("support custody needs an exact task plan")
        if type(self.execution_precommit) is not PanelFeatureExtractedExecutionPrecommit:
            raise TypeError("support custody needs an exact extracted precommit")
        if type(self.release_authorization) is not PanelFeatureExtractedReleaseAuthorization:
            raise TypeError("support custody needs an exact extracted authorization")
        if type(self.observer_matrix_artifact) is not CNNTypedAxisMatrixArtifact:
            raise TypeError("support custody rejects non-replayable observer artifacts")
        expected_ids = _expected_support_ids(self.task_plan)
        if (
            type(self.released_support_panels) is not tuple
            or type(self.released_support_store_receipts) is not tuple
            or len(self.released_support_panels) != 12
            or len(self.released_support_store_receipts) != 12
            or any(type(item) is not ReleasedOfficialExtractedPanel for item in self.released_support_panels)
            or tuple(item.panel_id for item in self.released_support_panels) != expected_ids
            or self.execution_precommit.targeted_drill_plan_digest
            != self.release_authorization.targeted_drill_plan_digest
            or self.execution_precommit.record_digest
            != self.release_authorization.execution_precommit_digest
            or self.task_plan.task_id not in self.release_authorization.selected_task_ids
            or not set(expected_ids) <= set(self.release_authorization.authorized_support_panel_ids)
            or set(self.sealed_query_panel_ids)
            != {self.task_plan.side_1_query_panel_id, self.task_plan.side_0_query_panel_id}
            or not set(self.sealed_query_panel_ids)
            <= set(self.release_authorization.sealed_query_panel_ids)
        ):
            raise TypedAxisCustodyV2Error("support task, release, or role lineage differs")
        rows = self.observer_matrix_artifact.prediction_batch.rows
        grant = self.observer_matrix_artifact.population_grant
        target_binding = (
            grant.target_release_authorization_address
            if grant.target_release_authorization_address is not None
            else self.observer_matrix_artifact.prediction_batch.target_authorization_record_digest
        )
        if (
            self.observer_matrix_artifact.prediction_batch.task_id != self.task_plan.task_id
            or tuple(row.panel_id for row in rows) != expected_ids
            or target_binding != self.release_authorization.record_digest
        ):
            raise TypedAxisCustodyV2Error("observer matrix belongs to another task or release")
        for released, receipt, prediction in zip(
            self.released_support_panels,
            self.released_support_store_receipts,
            rows,
            strict=True,
        ):
            _receipt_binds(receipt, released, object_kind="released-extracted-support-panel")
            if (
                released.exact_png_digest != prediction.png_sha256
                or len(released.exact_png_bytes) != prediction.png_size_bytes
                or released.execution_precommit_digest != self.execution_precommit.record_digest
                or released.exposure_successor_digest
                != self.release_authorization.exposure_successor_digest
                or released.release_receipt.extracted_archive_digest
                != self.release_authorization.extracted_archive_record_digest
                or released.release_receipt.corpus_manifest_digest
                != self.release_authorization.extracted_corpus_manifest_digest
                or released.release_receipt.release_descriptor_digest
                != self.release_authorization.release_descriptor_digest
            ):
                raise TypedAxisCustodyV2Error("support observer bytes differ from release custody")
        _address(self.record_digest, "support custody digest")
        if self.record_digest != "sha256:" + canonical_digest(_support_content(self)):
            raise TypedAxisCustodyV2Error("support custody digest differs")

    @property
    def task_id(self) -> str:
        return self.task_plan.task_id

    @property
    def matrix(self) -> TypedSupportMatrix:
        return self.observer_matrix_artifact.matrix

    @property
    def artifact_address(self) -> str:
        return self.record_digest

    @property
    def sealed_query_panel_ids(self) -> tuple[str, str]:
        return (self.task_plan.side_0_query_panel_id, self.task_plan.side_1_query_panel_id)

    @classmethod
    def create(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit: PanelFeatureExtractedExecutionPrecommit,
        release_authorization: PanelFeatureExtractedReleaseAuthorization,
        released_support_panels: Sequence[ReleasedOfficialExtractedPanel],
        released_support_store_receipts: Sequence[ObjectBongardWriteOnceReceipt],
        observer_matrix_artifact: CNNTypedAxisMatrixArtifact,
    ) -> "TaskBoundTypedAxisSupportArtifact":
        values = {
            "task_plan": task_plan,
            "execution_precommit": execution_precommit,
            "release_authorization": release_authorization,
            "released_support_panels": tuple(released_support_panels),
            "released_support_store_receipts": tuple(released_support_store_receipts),
            "observer_matrix_artifact": observer_matrix_artifact,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest="sha256:" + canonical_digest(_support_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_support_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TaskBoundTypedAxisSupportArtifact":
        expected = {
            "schema", "source_digest", "task_plan", "task_plan_digest",
            "execution_precommit", "execution_precommit_digest",
            "release_authorization", "release_authorization_digest",
            "released_support_panels", "released_support_store_receipts",
            "observer_matrix_artifact", "observer_matrix_artifact_address",
            "matrix_address", "support_role_order", "support_panel_count",
            "sealed_query_panel_ids", "query_panel_count",
            "support_pixels_and_observations_bound_before_inventory",
            "opaque_observer_artifacts_accepted", "training_supervision_artifacts_accepted",
            "observer_inference_externally_authenticated", "benchmark_sealable",
            "query_release_authorized",
            "query_pixels_seen", "python_is_canonical_authority", "lean_present",
            "record_digest",
        }
        raw = _fields(value, expected, "typed-axis support custody")
        if (
            raw["schema"] != SUPPORT_CUSTODY_SCHEMA
            or raw["source_digest"] != panel_typed_axis_custody_v2_source_digest()
            or raw["support_role_order"]
            != "task-side0-folder1-primary_then-task-side1-folder0-contrast"
            or raw["support_panel_count"] != 12
            or raw["query_panel_count"] != 0
            or raw["support_pixels_and_observations_bound_before_inventory"] is not True
            or raw["opaque_observer_artifacts_accepted"] is not False
            or raw["training_supervision_artifacts_accepted"] is not False
            or raw["observer_inference_externally_authenticated"] is not False
            or raw["benchmark_sealable"] is not False
            or raw["query_release_authorized"] is not False
            or raw["query_pixels_seen"] is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or type(raw["released_support_panels"]) is not list
            or type(raw["released_support_store_receipts"]) is not list
        ):
            raise TypedAxisCustodyV2Error("typed-axis support custody policy differs")
        result = cls(
            ObjectBongardTaskPlan.from_data(raw["task_plan"]),
            PanelFeatureExtractedExecutionPrecommit.from_data(raw["execution_precommit"]),
            PanelFeatureExtractedReleaseAuthorization.from_data(raw["release_authorization"]),
            tuple(ReleasedOfficialExtractedPanel.from_data(item) for item in raw["released_support_panels"]),
            tuple(ObjectBongardWriteOnceReceipt.from_data(item) for item in raw["released_support_store_receipts"]),
            CNNTypedAxisMatrixArtifact.from_data(raw["observer_matrix_artifact"]),
            raw["record_digest"],
        )
        if (
            raw["task_plan_digest"] != result.task_plan.record_digest
            or raw["execution_precommit_digest"] != result.execution_precommit.record_digest
            or raw["release_authorization_digest"] != result.release_authorization.record_digest
            or raw["observer_matrix_artifact_address"] != result.observer_matrix_artifact.artifact_address
            or raw["matrix_address"] != result.matrix.matrix_address
            or raw["sealed_query_panel_ids"] != list(result.sealed_query_panel_ids)
            or result.to_data() != dict(raw)
        ):
            raise TypedAxisCustodyV2Error("typed-axis support custody is not canonical")
        return result


def cold_replay_task_bound_typed_axis_support(
    artifact: TaskBoundTypedAxisSupportArtifact,
    *,
    store: ObjectBongardReleaseStore,
    archive: OfficialExtractedPanelArchive,
    expected_artifact_address: str,
) -> TaskBoundTypedAxisSupportArtifact:
    if type(artifact) is not TaskBoundTypedAxisSupportArtifact:
        raise TypeError("support replay needs exact TaskBoundTypedAxisSupportArtifact")
    expected = _address(expected_artifact_address, "expected support custody")
    restored = TaskBoundTypedAxisSupportArtifact.from_data(artifact.to_data())
    if restored.record_digest != expected:
        raise TypedAxisCustodyV2Error("support custody differs from external commitment")
    cold_replay_cnn_typed_support_matrix(
        restored.observer_matrix_artifact,
        expected_artifact_address=restored.observer_matrix_artifact.artifact_address,
    )
    for released, receipt in zip(
        restored.released_support_panels,
        restored.released_support_store_receipts,
        strict=True,
    ):
        store.verify(receipt, expected_data=released.to_data())
        released.cold_verify(
            archive,
            expected_execution_precommit_digest=restored.execution_precommit.record_digest,
            expected_exposure_successor_digest=restored.release_authorization.exposure_successor_digest,
        )
    return restored


def _query_cells(
    prediction: SupportPanelPrediction,
    *,
    observer_protocol_digest: str,
    population_grant: CNNPopulationGrant,
) -> tuple[TypedAxisCell, ...]:
    values: list[TypedAxisCell] = []
    for axis in AXES:
        if axis is Axis.STRAIGHT_ACTION_COUNT:
            if prediction.straight_class_set:
                cell = TypedAxisCell.calibrated_set(
                    axis,
                    prediction.straight_class_set,
                    observer_protocol_digest,
                    population_grant.grant_address,
                )
            else:
                cell = TypedAxisCell.error(axis, observer_protocol_digest, "empty_straight_class_set")
        elif axis is Axis.CATALOG_CONVEXITY:
            raw = prediction.catalog_class_set
            if not raw:
                cell = TypedAxisCell.error(axis, observer_protocol_digest, "empty_catalog_class_set")
            elif 0 in raw:
                cell = TypedAxisCell.gap(axis, observer_protocol_digest, "catalog_set_contains_unresolved")
            else:
                mapped = tuple("catalog_nonconvex" if item == 1 else "catalog_convex" for item in raw)
                cell = TypedAxisCell.calibrated_set(
                    axis, mapped, observer_protocol_digest, population_grant.grant_address
                )
        else:
            cell = TypedAxisCell.gap(axis, observer_protocol_digest, "cnn_axis_not_observed")
        values.append(cell)
    return tuple(values)


def _query_content(value: "TypedAxisQueryObservationArtifact") -> dict[str, object]:
    return {
        "schema": QUERY_OBSERVATION_SCHEMA,
        "source_digest": panel_typed_axis_custody_v2_source_digest(),
        "support_custody_address": value.support_custody.record_digest,
        "formula_commit_address": value.formula_commit_address,
        "released_query_panel": value.released_query_panel.to_data(),
        "released_query_store_receipt": value.released_query_store_receipt.to_data(),
        "prediction": value.prediction.to_data(),
        "observer_protocol_digest": value.observer_protocol_digest,
        "population_grant_address": value.support_custody.observer_matrix_artifact.population_grant.grant_address,
        "cells": [item.to_data() for item in value.cells],
        "query_truth_label_present": False,
        "formula_or_selected_predicate_passed_to_observer": False,
        "query_released_only_after_formula_commit": True,
        "observer_projection_replayed_from_archived_probabilities": True,
        "python_is_canonical_authority": True,
        "lean_present": False,
    }


@dataclass(frozen=True, slots=True)
class TypedAxisQueryObservationArtifact:
    """One released query PNG and its candidate-independent typed cells."""

    support_custody: TaskBoundTypedAxisSupportArtifact
    formula_commit_address: str
    released_query_panel: ReleasedOfficialExtractedPanel
    released_query_store_receipt: ObjectBongardWriteOnceReceipt
    prediction: SupportPanelPrediction
    cells: tuple[TypedAxisCell, ...]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.support_custody) is not TaskBoundTypedAxisSupportArtifact:
            raise TypeError("query observation needs exact support custody")
        _address(self.formula_commit_address, "query formula commit")
        if type(self.released_query_panel) is not ReleasedOfficialExtractedPanel:
            raise TypeError("query observation needs an exact released query panel")
        if type(self.prediction) is not SupportPanelPrediction:
            raise TypeError("query observation needs an exact frozen prediction")
        _receipt_binds(
            self.released_query_store_receipt,
            self.released_query_panel,
            object_kind="released-extracted-query-panel",
        )
        expected_side = (
            SupportSide.PRIMARY
            if self.released_query_panel.panel_id
            == self.support_custody.task_plan.side_0_query_panel_id
            else SupportSide.CONTRAST
        )
        if (
            self.released_query_panel.panel_id not in self.support_custody.sealed_query_panel_ids
            or self.prediction.panel_id != self.released_query_panel.panel_id
            or self.prediction.side is not expected_side
            or self.prediction.ordinal != QUERY_ORDINAL
            or self.prediction.task_id != self.support_custody.task_id
            or self.prediction.png_sha256 != self.released_query_panel.exact_png_digest
            or self.prediction.png_size_bytes != len(self.released_query_panel.exact_png_bytes)
            or self.released_query_panel.execution_precommit_digest
            != self.support_custody.execution_precommit.record_digest
            or self.released_query_panel.exposure_successor_digest
            != self.support_custody.release_authorization.exposure_successor_digest
        ):
            raise TypedAxisCustodyV2Error("query prediction differs from released query custody")
        grant = self.support_custody.observer_matrix_artifact.population_grant
        grant.authorize_task(self.support_custody.task_id)
        self.prediction.verify_q(grant.deployment_joint_q)
        expected_cells = _query_cells(
            self.prediction,
            observer_protocol_digest=self.observer_protocol_digest,
            population_grant=grant,
        )
        if type(self.cells) is not tuple or self.cells != expected_cells:
            raise TypedAxisCustodyV2Error("query typed cells differ from deterministic projection")
        _address(self.record_digest, "query observation digest")
        if self.record_digest != "sha256:" + canonical_digest(_query_content(self)):
            raise TypedAxisCustodyV2Error("query observation digest differs")

    @property
    def observer_protocol_digest(self) -> str:
        return self.support_custody.matrix.observer_protocol_digest

    @classmethod
    def create(
        cls,
        *,
        support_custody: TaskBoundTypedAxisSupportArtifact,
        formula_commit_address: str,
        released_query_panel: ReleasedOfficialExtractedPanel,
        released_query_store_receipt: ObjectBongardWriteOnceReceipt,
        prediction: SupportPanelPrediction,
    ) -> "TypedAxisQueryObservationArtifact":
        cells = _query_cells(
            prediction,
            observer_protocol_digest=support_custody.matrix.observer_protocol_digest,
            population_grant=support_custody.observer_matrix_artifact.population_grant,
        )
        values = {
            "support_custody": support_custody,
            "formula_commit_address": formula_commit_address,
            "released_query_panel": released_query_panel,
            "released_query_store_receipt": released_query_store_receipt,
            "prediction": prediction,
            "cells": cells,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest="sha256:" + canonical_digest(_query_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_query_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls,
        value: object,
        *,
        support_custody: TaskBoundTypedAxisSupportArtifact,
    ) -> "TypedAxisQueryObservationArtifact":
        expected = {
            "schema", "source_digest", "support_custody_address", "formula_commit_address",
            "released_query_panel", "released_query_store_receipt", "prediction",
            "observer_protocol_digest", "population_grant_address", "cells",
            "query_truth_label_present", "formula_or_selected_predicate_passed_to_observer",
            "query_released_only_after_formula_commit",
            "observer_projection_replayed_from_archived_probabilities",
            "python_is_canonical_authority", "lean_present", "record_digest",
        }
        raw = _fields(value, expected, "typed-axis query observation")
        if (
            raw["schema"] != QUERY_OBSERVATION_SCHEMA
            or raw["source_digest"] != panel_typed_axis_custody_v2_source_digest()
            or raw["support_custody_address"] != support_custody.record_digest
            or raw["observer_protocol_digest"] != support_custody.matrix.observer_protocol_digest
            or raw["population_grant_address"]
            != support_custody.observer_matrix_artifact.population_grant.grant_address
            or raw["query_truth_label_present"] is not False
            or raw["formula_or_selected_predicate_passed_to_observer"] is not False
            or raw["query_released_only_after_formula_commit"] is not True
            or raw["observer_projection_replayed_from_archived_probabilities"] is not True
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or type(raw["cells"]) is not list
        ):
            raise TypedAxisCustodyV2Error("typed-axis query observation policy differs")
        result = cls(
            support_custody,
            raw["formula_commit_address"],
            ReleasedOfficialExtractedPanel.from_data(raw["released_query_panel"]),
            ObjectBongardWriteOnceReceipt.from_data(raw["released_query_store_receipt"]),
            SupportPanelPrediction.from_data(raw["prediction"]),
            tuple(TypedAxisCell.from_data(item) for item in raw["cells"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise TypedAxisCustodyV2Error("typed-axis query observation is not canonical")
        return result


def cold_replay_typed_axis_query_observation(
    artifact: TypedAxisQueryObservationArtifact,
    *,
    store: ObjectBongardReleaseStore,
    archive: OfficialExtractedPanelArchive,
    expected_artifact_address: str,
) -> TypedAxisQueryObservationArtifact:
    if type(artifact) is not TypedAxisQueryObservationArtifact:
        raise TypeError("query replay needs exact TypedAxisQueryObservationArtifact")
    expected = _address(expected_artifact_address, "expected query observation")
    support = cold_replay_task_bound_typed_axis_support(
        artifact.support_custody,
        store=store,
        archive=archive,
        expected_artifact_address=artifact.support_custody.record_digest,
    )
    restored = TypedAxisQueryObservationArtifact.from_data(
        artifact.to_data(), support_custody=support
    )
    if restored.record_digest != expected:
        raise TypedAxisCustodyV2Error("query observation differs from external commitment")
    store.verify(
        restored.released_query_store_receipt,
        expected_data=restored.released_query_panel.to_data(),
    )
    restored.released_query_panel.cold_verify(
        archive,
        expected_execution_precommit_digest=support.execution_precommit.record_digest,
        expected_exposure_successor_digest=support.release_authorization.exposure_successor_digest,
    )
    return restored


__all__ = (
    "QUERY_OBSERVATION_SCHEMA",
    "SUPPORT_CUSTODY_SCHEMA",
    "TaskBoundTypedAxisSupportArtifact",
    "TypedAxisCustodyV2Error",
    "TypedAxisQueryObservationArtifact",
    "cold_replay_task_bound_typed_axis_support",
    "cold_replay_typed_axis_query_observation",
    "panel_typed_axis_custody_v2_source_digest",
)
