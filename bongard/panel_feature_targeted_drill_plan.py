"""Metadata-only targeted TRAIN drills for one already disclosed task family.

The broad batch planners deliberately spread work across dataset families.  A
representation drill has a different purpose: after one generator semantic has
already been disclosed, exercise new predicate machinery on exact-image-unused
siblings without spending a semantics-fresh reserve.  This module makes that
choice explicit and reproducible while sealing both query identities before a
panel can be released.

Only official task IDs, split membership, an exposure ledger, and a text seed
enter selection.  Pixel paths, PNG bytes, action programs, and labels inferred
from image content are not accepted.  Python is the canonical plan authority;
Lean is absent and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    ObjectBongardTaskPlan,
    object_bongard_task_inventory_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


TARGETED_DRILL_PLAN_SCHEMA = "gkm.bongard-panel-feature-targeted-drill-plan.v1"
TARGETED_DRILL_ALGORITHM_ID = (
    "bongard.panel-feature/metadata-only-exact-unused-semantic-reuse-v1"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_INSTANCE = re.compile(
    r"(?P<semantic>(?:bd|ff|hd)_[A-Za-z0-9_.-]+)_(?P<index>[0-9]{4})\Z"
)
_SEMANTIC_KEY = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")


class PanelFeatureTargetedDrillPlanError(ValueError):
    """A metadata input, semantic-reuse proof, or sealed plan differs."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelFeatureTargetedDrillPlanError(
            f"{label} must be a sha256: address"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureTargetedDrillPlanError(f"{label} fields differ")
    return value


def _frozen_task_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PanelFeatureTargetedDrillPlanError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        result != tuple(sorted(set(result)))
        or any(type(item) is not str or _TASK_INSTANCE.fullmatch(item) is None for item in result)
    ):
        raise PanelFeatureTargetedDrillPlanError(
            f"{label} must contain canonical unique sorted official task IDs"
        )
    return result


def _semantic_key(value: object) -> str:
    if (
        type(value) is not str
        or _SEMANTIC_KEY.fullmatch(value) is None
        or _TASK_INSTANCE.fullmatch(value) is not None
    ):
        raise PanelFeatureTargetedDrillPlanError(
            "target semantic key must omit the terminal four-digit task instance"
        )
    return value


def _task_semantic_key(task_id: str) -> str:
    match = _TASK_INSTANCE.fullmatch(task_id)
    if match is None:
        raise PanelFeatureTargetedDrillPlanError("task ID has no semantic-instance split")
    return match.group("semantic")


def _selection_seed(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise PanelFeatureTargetedDrillPlanError("selection seed is invalid")
    return value


def panel_feature_targeted_drill_plan_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_selection_release_or_replay": False,
    }


def panel_feature_targeted_drill_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-panel-feature-targeted-drill-algorithm.v1",
            "algorithm_id": TARGETED_DRILL_ALGORITHM_ID,
            "source_sha256": panel_feature_targeted_drill_plan_source_digest(),
            "selection": "hash-rank-exact-unused-in-one-disclosed-semantic-family",
            "support_query_split": "six-support-one-query-per-physical-side",
            "query_identities_sealed_before_pixels": True,
            **_authority_data(),
        }
    )


def _rank(seed_digest: str, task_id: str) -> str:
    return canonical_digest(
        {
            "algorithm_id": TARGETED_DRILL_ALGORITHM_ID,
            "selection_seed_digest": seed_digest,
            "domain": "target-semantic-task-instance",
            "task_id": task_id,
        }
    )


def _plan_content(value: "PanelFeatureTargetedDrillPlan") -> dict[str, object]:
    return {
        "schema": TARGETED_DRILL_PLAN_SCHEMA,
        "algorithm_id": TARGETED_DRILL_ALGORITHM_ID,
        "algorithm_digest": panel_feature_targeted_drill_algorithm_digest(),
        "algorithm_source_sha256": panel_feature_targeted_drill_plan_source_digest(),
        "selection_seed_digest": value.selection_seed_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "split_source_digest": value.split_source_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "train_task_ids_digest": value.train_task_ids_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "exposed_task_ids_digest": value.exposed_task_ids_digest,
        "target_semantic_key": value.target_semantic_key,
        "semantic_reuse_witness_task_ids": list(value.semantic_reuse_witness_task_ids),
        "semantic_reuse_witness_digest": _address(
            list(value.semantic_reuse_witness_task_ids)
        ),
        "exact_unused_candidate_task_ids_digest": (
            value.exact_unused_candidate_task_ids_digest
        ),
        "exact_unused_candidate_count": value.exact_unused_candidate_count,
        "requested_task_count": value.requested_task_count,
        "selection_order_task_ids": list(value.selection_order_task_ids),
        "selection_order_task_ids_digest": _address(
            list(value.selection_order_task_ids)
        ),
        "tasks": [item.to_data() for item in value.tasks],
        "selected_task_ids_digest": _address([item.task_id for item in value.tasks]),
        "sealed_query_panel_ids_digest": _address(
            sorted(
                panel_id
                for item in value.tasks
                for panel_id in (
                    item.side_0_query_panel_id,
                    item.side_1_query_panel_id,
                )
            )
        ),
        "selection_inputs_include_pixels": False,
        "selection_inputs_include_pixel_paths": False,
        "selection_inputs_include_action_programs": False,
        "panel_bytes_opened_during_selection": False,
        "selected_tasks_exact_image_unused": True,
        "selected_semantics_previously_disclosed": True,
        "semantics_fresh_claim_authorized": False,
        "official_test_authorized": False,
        "query_identities_sealed_before_support_pixels": True,
        "claim": "targeted-train-engineering-drill-not-official-benchmark",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTargetedDrillPlan:
    selection_seed_digest: str
    release_descriptor_digest: str
    split_source_digest: str
    task_inventory_digest: str
    train_task_ids_digest: str
    exposure_predecessor_digest: str
    exposed_task_ids_digest: str
    target_semantic_key: str
    semantic_reuse_witness_task_ids: tuple[str, ...]
    exact_unused_candidate_task_ids_digest: str
    exact_unused_candidate_count: int
    requested_task_count: int
    selection_order_task_ids: tuple[str, ...]
    tasks: tuple[ObjectBongardTaskPlan, ...]
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "selection_seed_digest",
            "release_descriptor_digest",
            "split_source_digest",
            "task_inventory_digest",
            "train_task_ids_digest",
            "exposure_predecessor_digest",
            "exposed_task_ids_digest",
            "exact_unused_candidate_task_ids_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        semantic = _semantic_key(self.target_semantic_key)
        if (
            type(self.semantic_reuse_witness_task_ids) is not tuple
            or not self.semantic_reuse_witness_task_ids
            or self.semantic_reuse_witness_task_ids
            != tuple(sorted(set(self.semantic_reuse_witness_task_ids)))
            or any(
                _task_semantic_key(item) != semantic
                for item in self.semantic_reuse_witness_task_ids
            )
            or type(self.exact_unused_candidate_count) is not int
            or type(self.requested_task_count) is not int
            or self.requested_task_count <= 0
            or self.exact_unused_candidate_count < self.requested_task_count
            or type(self.selection_order_task_ids) is not tuple
            or len(self.selection_order_task_ids) != self.requested_task_count
            or len(set(self.selection_order_task_ids)) != self.requested_task_count
            or any(
                _task_semantic_key(item) != semantic
                or item in self.semantic_reuse_witness_task_ids
                for item in self.selection_order_task_ids
            )
            or type(self.tasks) is not tuple
            or len(self.tasks) != self.requested_task_count
            or any(type(item) is not ObjectBongardTaskPlan for item in self.tasks)
            or tuple(item.task_id for item in self.tasks)
            != tuple(sorted(self.selection_order_task_ids))
            or any(item.split != "train" for item in self.tasks)
            or self.record_digest != _address(_plan_content(self))
        ):
            raise PanelFeatureTargetedDrillPlanError(
                "targeted drill plan identity differs"
            )

    @property
    def algorithm_digest(self) -> str:
        return panel_feature_targeted_drill_algorithm_digest()

    @property
    def source_digest(self) -> str:
        return panel_feature_targeted_drill_plan_source_digest()

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTargetedDrillPlan":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "algorithm_source_sha256",
                "selection_seed_digest",
                "release_descriptor_digest",
                "split_source_digest",
                "task_inventory_digest",
                "train_task_ids_digest",
                "exposure_predecessor_digest",
                "exposed_task_ids_digest",
                "target_semantic_key",
                "semantic_reuse_witness_task_ids",
                "semantic_reuse_witness_digest",
                "exact_unused_candidate_task_ids_digest",
                "exact_unused_candidate_count",
                "requested_task_count",
                "selection_order_task_ids",
                "selection_order_task_ids_digest",
                "tasks",
                "selected_task_ids_digest",
                "sealed_query_panel_ids_digest",
                "selection_inputs_include_pixels",
                "selection_inputs_include_pixel_paths",
                "selection_inputs_include_action_programs",
                "panel_bytes_opened_during_selection",
                "selected_tasks_exact_image_unused",
                "selected_semantics_previously_disclosed",
                "semantics_fresh_claim_authorized",
                "official_test_authorized",
                "query_identities_sealed_before_support_pixels",
                "claim",
                *_authority_data(),
                "record_digest",
            },
            "targeted panel-feature drill plan",
        )
        if (
            raw["schema"] != TARGETED_DRILL_PLAN_SCHEMA
            or raw["algorithm_id"] != TARGETED_DRILL_ALGORITHM_ID
            or raw["algorithm_digest"]
            != panel_feature_targeted_drill_algorithm_digest()
            or raw["algorithm_source_sha256"]
            != panel_feature_targeted_drill_plan_source_digest()
            or raw["selection_inputs_include_pixels"] is not False
            or raw["selection_inputs_include_pixel_paths"] is not False
            or raw["selection_inputs_include_action_programs"] is not False
            or raw["panel_bytes_opened_during_selection"] is not False
            or raw["selected_tasks_exact_image_unused"] is not True
            or raw["selected_semantics_previously_disclosed"] is not True
            or raw["semantics_fresh_claim_authorized"] is not False
            or raw["official_test_authorized"] is not False
            or raw["query_identities_sealed_before_support_pixels"] is not True
            or raw["claim"]
            != "targeted-train-engineering-drill-not-official-benchmark"
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["semantic_reuse_witness_task_ids"]) is not list
            or type(raw["selection_order_task_ids"]) is not list
            or type(raw["tasks"]) is not list
        ):
            raise PanelFeatureTargetedDrillPlanError(
                "targeted drill plan policy differs"
            )
        result = cls(
            raw["selection_seed_digest"],
            raw["release_descriptor_digest"],
            raw["split_source_digest"],
            raw["task_inventory_digest"],
            raw["train_task_ids_digest"],
            raw["exposure_predecessor_digest"],
            raw["exposed_task_ids_digest"],
            raw["target_semantic_key"],
            tuple(raw["semantic_reuse_witness_task_ids"]),
            raw["exact_unused_candidate_task_ids_digest"],
            raw["exact_unused_candidate_count"],
            raw["requested_task_count"],
            tuple(raw["selection_order_task_ids"]),
            tuple(ObjectBongardTaskPlan.from_data(item) for item in raw["tasks"]),
            raw["record_digest"],
        )
        content = _plan_content(result)
        for key in (
            "semantic_reuse_witness_digest",
            "selection_order_task_ids_digest",
            "selected_task_ids_digest",
            "sealed_query_panel_ids_digest",
        ):
            if raw[key] != content[key]:
                raise PanelFeatureTargetedDrillPlanError(
                    f"targeted drill derived {key} differs"
                )
        if result.to_data() != dict(raw):
            raise PanelFeatureTargetedDrillPlanError(
                "targeted drill plan is not canonical"
            )
        return result


def plan_panel_feature_targeted_drill(
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    predecessor: ExposureLedger,
    target_semantic_key: str,
    selection_seed: str,
    requested_task_count: int,
    release_descriptor_digest: str,
    split_source_digest: str,
    task_inventory_digest: str,
) -> PanelFeatureTargetedDrillPlan:
    """Select exact-unused siblings of one already disclosed semantic family."""

    if type(predecessor) is not ExposureLedger:
        raise TypeError("predecessor must be exact ExposureLedger")
    inventory = _frozen_task_ids(task_ids, "task inventory")
    train = _frozen_task_ids(train_task_ids, "TRAIN task inventory")
    if not set(train) <= set(inventory):
        raise PanelFeatureTargetedDrillPlanError(
            "TRAIN inventory is outside the official task inventory"
        )
    semantic = _semantic_key(target_semantic_key)
    seed = _selection_seed(selection_seed)
    if (
        type(requested_task_count) is not int
        or requested_task_count <= 0
    ):
        raise PanelFeatureTargetedDrillPlanError(
            "requested task count must be a positive exact integer"
        )
    for value, label in (
        (release_descriptor_digest, "release descriptor"),
        (split_source_digest, "split source"),
        (task_inventory_digest, "task inventory"),
    ):
        _require_address(value, label)
    if task_inventory_digest != object_bongard_task_inventory_digest(inventory):
        raise PanelFeatureTargetedDrillPlanError(
            "task inventory differs from its official commitment"
        )
    exposed = tuple(sorted(predecessor.exposed_task_ids))
    if not set(exposed) <= set(inventory):
        raise PanelFeatureTargetedDrillPlanError(
            "exposure predecessor names a task outside the official inventory"
        )
    witnesses = tuple(
        item for item in exposed if _task_semantic_key(item) == semantic
    )
    if not witnesses:
        raise PanelFeatureTargetedDrillPlanError(
            "targeted semantics have not previously been disclosed"
        )
    candidates = tuple(
        item
        for item in train
        if item not in predecessor.exposed_task_ids
        and _task_semantic_key(item) == semantic
    )
    if len(candidates) < requested_task_count:
        raise PanelFeatureTargetedDrillPlanError(
            "target semantic family has too few exact-unused TRAIN instances"
        )
    seed_digest = "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()
    selected = tuple(
        sorted(candidates, key=lambda item: (_rank(seed_digest, item), item))[
            :requested_task_count
        ]
    )
    tasks = tuple(
        sorted(
            (
                ObjectBongardTaskPlan.create(item, seed_digest=seed_digest)
                for item in selected
            ),
            key=lambda item: item.task_id,
        )
    )
    values: dict[str, object] = {
        "selection_seed_digest": seed_digest,
        "release_descriptor_digest": release_descriptor_digest,
        "split_source_digest": split_source_digest,
        "task_inventory_digest": task_inventory_digest,
        "train_task_ids_digest": _address(list(train)),
        "exposure_predecessor_digest": predecessor.digest,
        "exposed_task_ids_digest": _address(list(exposed)),
        "target_semantic_key": semantic,
        "semantic_reuse_witness_task_ids": witnesses,
        "exact_unused_candidate_task_ids_digest": _address(list(candidates)),
        "exact_unused_candidate_count": len(candidates),
        "requested_task_count": requested_task_count,
        "selection_order_task_ids": selected,
        "tasks": tasks,
    }
    provisional = object.__new__(PanelFeatureTargetedDrillPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelFeatureTargetedDrillPlan(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_plan_content(provisional)),
    )


def verify_panel_feature_targeted_drill_plan(
    plan: PanelFeatureTargetedDrillPlan,
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    predecessor: ExposureLedger,
    selection_seed: str,
) -> PanelFeatureTargetedDrillPlan:
    """Cold-reproduce a targeted plan from the same metadata-only inputs."""

    if type(plan) is not PanelFeatureTargetedDrillPlan:
        raise TypeError("plan must be exact PanelFeatureTargetedDrillPlan")
    replay = plan_panel_feature_targeted_drill(
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        predecessor=predecessor,
        target_semantic_key=plan.target_semantic_key,
        selection_seed=selection_seed,
        requested_task_count=plan.requested_task_count,
        release_descriptor_digest=plan.release_descriptor_digest,
        split_source_digest=plan.split_source_digest,
        task_inventory_digest=plan.task_inventory_digest,
    )
    if replay != plan:
        raise PanelFeatureTargetedDrillPlanError(
            "targeted drill plan differs from metadata replay"
        )
    return plan


__all__ = (
    "TARGETED_DRILL_ALGORITHM_ID",
    "TARGETED_DRILL_PLAN_SCHEMA",
    "PanelFeatureTargetedDrillPlan",
    "PanelFeatureTargetedDrillPlanError",
    "panel_feature_targeted_drill_algorithm_digest",
    "panel_feature_targeted_drill_plan_source_digest",
    "plan_panel_feature_targeted_drill",
    "verify_panel_feature_targeted_drill_plan",
)
