"""Bind a closed-catalog support inventory to exact task evidence custody.

``ClosedCatalogSupportInventory`` intentionally contains no corpus panel IDs;
by itself it cannot prove that its first six observations came from side 0 of
the task being frozen.  This module closes that boundary.  It accepts only an
exact task plan, an exact full-receipt evidence bundle, and the exact derived
inventory.  The support partition is then reconstructed from evidence panel
IDs and compared byte-for-byte with both the task plan and the inventory.

The schema has two explicit evidence branches: the legacy receipted
``PanelFeatureEvidenceBundle`` and the hierarchical macro/micro
``HierarchicalPanelFeatureEvidenceBundle``.  Structural lookalikes are never
accepted.  Cold replay verifies every archived receipt without invoking a
model boundary.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogSupportInventory,
    ProposerNarrationSnapshot,
    cold_replay_closed_catalog_support_inventory,
)
from bongard.panel_feature_evidence_bundle import (
    PanelFeatureEvidenceBundle,
    PanelFeatureEvidencePanel,
    PanelFeatureEvidencePhase,
    cold_replay_panel_feature_evidence_bundle,
)
from bongard.panel_hierarchical_feature_evidence_bundle import (
    HierarchicalFeatureEvidencePhase,
    HierarchicalPanelFeatureEvidenceBundle,
    HierarchicalPanelFeatureEvidenceRow,
    cold_replay_hierarchical_panel_feature_evidence_bundle,
    verified_hierarchical_observation_sets,
)
from bongard.panel_feature_observation import PanelFeatureObservationSet
from bongard.panel_soft_ontology import NativeOrientation
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


TASK_BOUND_CLOSED_CATALOG_INVENTORY_SCHEMA = (
    "gkm.bongard-task-bound-closed-catalog-support-inventory.v1"
)
TASK_BOUND_CLOSED_CATALOG_INVENTORY_ID = (
    "bongard.panel-feature/task-bound-closed-catalog-inventory-python-v1"
)
LEGACY_EVIDENCE_KIND = "legacy_full_receipt_panel_feature_evidence_v2"
HIERARCHICAL_EVIDENCE_KIND = (
    "hierarchical_full_receipt_panel_feature_evidence_v1"
)

EvidenceBundle = PanelFeatureEvidenceBundle | HierarchicalPanelFeatureEvidenceBundle
EvidenceRow = PanelFeatureEvidencePanel | HierarchicalPanelFeatureEvidenceRow

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TaskBoundClosedCatalogInventoryError(ValueError):
    """Task, panel-role evidence, or derived support inventory differs."""


def panel_feature_task_bound_inventory_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise TaskBoundClosedCatalogInventoryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise TaskBoundClosedCatalogInventoryError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TaskBoundClosedCatalogInventoryError(
            f"{label} must be a sha256: address"
        )
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "implementation_language": "python",
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "benchmark_authoritative": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_partition_selection_decision_or_replay": False,
    }


def _custody_policy_data() -> dict[str, object]:
    return {
        "support_partition_derived_from_exact_panel_evidence": True,
        "bare_observation_sequence_accepted": False,
        "structural_evidence_protocol_accepted": False,
        "accepted_exact_evidence_kinds": [
            LEGACY_EVIDENCE_KIND,
            HIERARCHICAL_EVIDENCE_KIND,
        ],
        "support_panel_count": 12,
        "query_panel_count": 0,
        "support_order": "task-side0-six-then-task-side1-six",
        "primary_orientation_fixed": NativeOrientation.SIDE0_POSITIVE.value,
        "opposite_orientation_diagnostic_only": True,
        "query_pixels_included": False,
        "query_observations_included": False,
        "live_model_calls_during_binding": 0,
        "cold_replay_model_calls": 0,
    }


def _canonical_task(value: object) -> ObjectBongardTaskPlan:
    if type(value) is not ObjectBongardTaskPlan:
        raise TypeError("task binding needs exact ObjectBongardTaskPlan")
    # Exact frozen values already ran their own complete constructor invariant.
    # Full serialization replay belongs at ``from_data``/cold-replay boundaries,
    # not at every parent construction.
    return value


def _canonical_evidence_bundle(value: object) -> EvidenceBundle:
    if type(value) is PanelFeatureEvidenceBundle:
        return value
    if type(value) is HierarchicalPanelFeatureEvidenceBundle:
        return value
    raise TypeError(
        "task binding needs one exact known full-receipt evidence bundle class"
    )


def _evidence_kind(value: EvidenceBundle) -> str:
    if type(value) is PanelFeatureEvidenceBundle:
        return LEGACY_EVIDENCE_KIND
    if type(value) is HierarchicalPanelFeatureEvidenceBundle:
        return HIERARCHICAL_EVIDENCE_KIND
    raise TypeError("task binding evidence class differs")


def _support_panels(
    value: EvidenceBundle,
) -> tuple[EvidenceRow, ...]:
    if type(value) is PanelFeatureEvidenceBundle:
        return value.panels_for_phase(PanelFeatureEvidencePhase.SUPPORT)
    if type(value) is HierarchicalPanelFeatureEvidenceBundle:
        return value.panels_for_phase(HierarchicalFeatureEvidencePhase.SUPPORT)
    raise TypeError("task binding evidence class differs")


def _query_panels(value: EvidenceBundle) -> tuple[EvidenceRow, ...]:
    if type(value) is PanelFeatureEvidenceBundle:
        return value.panels_for_phase(PanelFeatureEvidencePhase.QUERY)
    if type(value) is HierarchicalPanelFeatureEvidenceBundle:
        return value.panels_for_phase(HierarchicalFeatureEvidencePhase.QUERY)
    raise TypeError("task binding evidence class differs")


def _row_observation(value: EvidenceRow) -> PanelFeatureObservationSet:
    if type(value) is PanelFeatureEvidencePanel:
        return value.observation_set
    if type(value) is HierarchicalPanelFeatureEvidenceRow:
        return value.artifact.observation_set
    raise TypeError("task binding evidence row class differs")


def _verified_support_observations(
    value: EvidenceBundle,
) -> tuple[PanelFeatureObservationSet, ...]:
    if type(value) is PanelFeatureEvidenceBundle:
        return tuple(
            item.observation_set
            for item in value.panels_for_phase(PanelFeatureEvidencePhase.SUPPORT)
        )
    if type(value) is HierarchicalPanelFeatureEvidenceBundle:
        return verified_hierarchical_observation_sets(
            value,
            phase=HierarchicalFeatureEvidencePhase.SUPPORT,
            expected_bundle_address=value.bundle_address,
        )
    raise TypeError("task binding evidence class differs")


def _canonical_inventory(value: object) -> ClosedCatalogSupportInventory:
    if type(value) is not ClosedCatalogSupportInventory:
        raise TypeError("task binding needs exact ClosedCatalogSupportInventory")
    return value


def _support_ids(task: ObjectBongardTaskPlan) -> tuple[str, ...]:
    return task.side_0_support_panel_ids + task.side_1_support_panel_ids


def _query_ids(task: ObjectBongardTaskPlan) -> tuple[str, str]:
    return (task.side_0_query_panel_id, task.side_1_query_panel_id)


def _support_bindings(
    bundle: EvidenceBundle,
) -> tuple[tuple[str, str, str], ...]:
    panels = _support_panels(bundle)
    return tuple(
        (
            panel.panel_id,
            panel.panel_png_digest,
            _row_observation(panel).observation_set_digest,
        )
        for panel in panels
    )


def _bound_content(value: "TaskBoundClosedCatalogInventory") -> dict[str, object]:
    return {
        "schema": TASK_BOUND_CLOSED_CATALOG_INVENTORY_SCHEMA,
        "binding_id": TASK_BOUND_CLOSED_CATALOG_INVENTORY_ID,
        "binding_source_digest": panel_feature_task_bound_inventory_source_digest(),
        "task_plan": value.task_plan.to_data(),
        "task_plan_digest": value.task_plan.record_digest,
        "evidence_kind": _evidence_kind(value.evidence_bundle),
        "evidence_bundle": value.evidence_bundle.to_data(),
        "evidence_bundle_address": value.evidence_bundle.bundle_address,
        "closed_catalog_inventory": value.inventory.to_data(),
        "closed_catalog_inventory_address": value.inventory.artifact_address,
        "support_panel_bindings": [list(item) for item in value.support_panel_bindings],
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "observer_contract_digest": value.observer_contract_digest,
        "measurement_protocol_digest": value.measurement_protocol_digest,
        **_custody_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class TaskBoundClosedCatalogInventory:
    """Exact task/role/full-receipt custody for one closed support inventory."""

    task_plan: ObjectBongardTaskPlan
    evidence_bundle: EvidenceBundle
    inventory: ClosedCatalogSupportInventory
    support_panel_bindings: tuple[tuple[str, str, str], ...]
    sealed_query_panel_ids: tuple[str, str]
    observer_contract_digest: str
    measurement_protocol_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        task = _canonical_task(self.task_plan)
        bundle = _canonical_evidence_bundle(self.evidence_bundle)
        inventory = _canonical_inventory(self.inventory)
        support = _support_panels(bundle)
        query = _query_panels(bundle)
        expected_ids = _support_ids(task)
        expected_bindings = _support_bindings(bundle)
        expected_observations = _verified_support_observations(bundle)
        expected_axes = complete_whole_panel_feature_axes()
        narration = ProposerNarrationSnapshot.create(bundle.proposer_result)
        contracts = {item.observer_contract_digest for item in expected_observations}
        protocols = {
            item.measurement_protocol_digest for item in expected_observations
        }
        if (
            len(support) != 12
            or query
            or tuple(item.phase_index for item in support) != tuple(range(12))
            or tuple(item.panel_id for item in support) != expected_ids
            or tuple(item.panel_png_digest for item in support)
            != tuple(item.panel_digest for item in expected_observations)
            or bundle.observer_axes != expected_axes
            or any(
                tuple(item.axis for item in observation.axis_observations)
                != expected_axes
                for observation in expected_observations
            )
            or inventory.primary_orientation
            is not NativeOrientation.SIDE0_POSITIVE
            or inventory.support_observations != expected_observations
            or inventory.proposer_snapshot != narration
            or self.support_panel_bindings != expected_bindings
            or self.sealed_query_panel_ids != _query_ids(task)
            or set(expected_ids) & set(self.sealed_query_panel_ids)
            or len(contracts) != 1
            or len(protocols) != 1
            or self.observer_contract_digest != next(iter(contracts))
            or self.measurement_protocol_digest != next(iter(protocols))
        ):
            raise TaskBoundClosedCatalogInventoryError(
                "task, evidence support partition, proposer, or inventory differs"
            )
        _digest(self.observer_contract_digest, "observer contract digest")
        _digest(self.measurement_protocol_digest, "measurement protocol digest")
        _digest(self.record_digest, "task-bound inventory digest")
        if self.record_digest != canonical_digest(_bound_content(self)):
            raise TaskBoundClosedCatalogInventoryError(
                "task-bound inventory content address differs"
            )

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @classmethod
    def bind(
        cls,
        task_plan: ObjectBongardTaskPlan,
        evidence_bundle: EvidenceBundle,
        inventory: ClosedCatalogSupportInventory,
    ) -> "TaskBoundClosedCatalogInventory":
        task = _canonical_task(task_plan)
        bundle = _canonical_evidence_bundle(evidence_bundle)
        closed = _canonical_inventory(inventory)
        observations = tuple(
            _row_observation(item) for item in _support_panels(bundle)
        )
        if not observations:
            raise TaskBoundClosedCatalogInventoryError(
                "task-bound inventory has no support evidence"
            )
        values = {
            "task_plan": task,
            "evidence_bundle": bundle,
            "inventory": closed,
            "support_panel_bindings": _support_bindings(bundle),
            "sealed_query_panel_ids": _query_ids(task),
            "observer_contract_digest": observations[0].observer_contract_digest,
            "measurement_protocol_digest": observations[0].measurement_protocol_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_bound_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_bound_content(self), "record_digest": self.record_digest, "artifact_address": self.artifact_address}

    @classmethod
    def from_data(cls, value: object) -> "TaskBoundClosedCatalogInventory":
        raw = _fields(
            value,
            {
                "schema",
                "binding_id",
                "binding_source_digest",
                "task_plan",
                "task_plan_digest",
                "evidence_kind",
                "evidence_bundle",
                "evidence_bundle_address",
                "closed_catalog_inventory",
                "closed_catalog_inventory_address",
                "support_panel_bindings",
                "sealed_query_panel_ids",
                "observer_contract_digest",
                "measurement_protocol_digest",
                *_custody_policy_data(),
                *_authority_data(),
                "record_digest",
                "artifact_address",
            },
            "task-bound closed-catalog inventory",
        )
        policy = {**_custody_policy_data(), **_authority_data()}
        if (
            raw["schema"] != TASK_BOUND_CLOSED_CATALOG_INVENTORY_SCHEMA
            or raw["binding_id"] != TASK_BOUND_CLOSED_CATALOG_INVENTORY_ID
            or raw["binding_source_digest"]
            != panel_feature_task_bound_inventory_source_digest()
            or raw["evidence_kind"]
            not in (LEGACY_EVIDENCE_KIND, HIERARCHICAL_EVIDENCE_KIND)
            or any(raw[name] != item for name, item in policy.items())
            or type(raw["support_panel_bindings"]) is not list
            or type(raw["sealed_query_panel_ids"]) is not list
        ):
            raise TaskBoundClosedCatalogInventoryError(
                "task-bound inventory policy differs"
            )
        try:
            task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
            if raw["evidence_kind"] == LEGACY_EVIDENCE_KIND:
                bundle: EvidenceBundle = PanelFeatureEvidenceBundle.from_data(
                    raw["evidence_bundle"]
                )
            elif raw["evidence_kind"] == HIERARCHICAL_EVIDENCE_KIND:
                bundle = HierarchicalPanelFeatureEvidenceBundle.from_data(
                    raw["evidence_bundle"]
                )
            else:  # guarded above; kept explicit at the decoding boundary
                raise TaskBoundClosedCatalogInventoryError(
                    "task-bound inventory evidence kind differs"
                )
            inventory = ClosedCatalogSupportInventory.from_data(
                raw["closed_catalog_inventory"]
            )
            result = cls(
                task,
                bundle,
                inventory,
                tuple(tuple(item) for item in raw["support_panel_bindings"]),
                tuple(raw["sealed_query_panel_ids"]),
                raw["observer_contract_digest"],
                raw["measurement_protocol_digest"],
                raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TaskBoundClosedCatalogInventoryError):
                raise
            raise TaskBoundClosedCatalogInventoryError(
                "task-bound inventory value differs"
            ) from exc
        if (
            raw["task_plan_digest"] != task.record_digest
            or raw["evidence_bundle_address"] != bundle.bundle_address
            or raw["closed_catalog_inventory_address"] != inventory.artifact_address
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise TaskBoundClosedCatalogInventoryError(
                "task-bound inventory is not canonical"
            )
        return result


def cold_replay_task_bound_closed_catalog_inventory(
    archived: TaskBoundClosedCatalogInventory,
    *,
    expected_artifact_address: str,
) -> TaskBoundClosedCatalogInventory:
    """Replay task roles, full receipts, and inventory with zero model calls."""

    if type(archived) is not TaskBoundClosedCatalogInventory:
        raise TypeError("cold replay needs exact TaskBoundClosedCatalogInventory")
    expected = _address(expected_artifact_address, "expected task-bound address")
    restored = TaskBoundClosedCatalogInventory.from_data(archived.to_data())
    if type(restored.evidence_bundle) is PanelFeatureEvidenceBundle:
        cold_replay_panel_feature_evidence_bundle(
            restored.evidence_bundle,
            expected_bundle_address=restored.evidence_bundle.bundle_address,
        )
    elif type(restored.evidence_bundle) is HierarchicalPanelFeatureEvidenceBundle:
        cold_replay_hierarchical_panel_feature_evidence_bundle(
            restored.evidence_bundle,
            expected_bundle_address=restored.evidence_bundle.bundle_address,
        )
    else:  # exact union invariant; never accept a protocol lookalike
        raise TypeError("task-bound inventory evidence class differs")
    cold_replay_closed_catalog_support_inventory(
        restored.inventory,
        expected_artifact_address=restored.inventory.artifact_address,
    )
    replayed = TaskBoundClosedCatalogInventory.bind(
        restored.task_plan,
        restored.evidence_bundle,
        restored.inventory,
    )
    if replayed != archived or replayed.artifact_address != expected:
        raise TaskBoundClosedCatalogInventoryError(
            "task-bound inventory differs on cold replay"
        )
    return replayed


__all__ = (
    "HIERARCHICAL_EVIDENCE_KIND",
    "LEGACY_EVIDENCE_KIND",
    "TASK_BOUND_CLOSED_CATALOG_INVENTORY_ID",
    "TASK_BOUND_CLOSED_CATALOG_INVENTORY_SCHEMA",
    "TaskBoundClosedCatalogInventory",
    "TaskBoundClosedCatalogInventoryError",
    "cold_replay_task_bound_closed_catalog_inventory",
    "panel_feature_task_bound_inventory_source_digest",
)
