"""Metadata-only batch planning for the Python object-predicate pipeline.

The planner consumes authenticated task-ID sets, not corpus paths, images, or
action programs.  It selects an exact-unused TRAIN cohort across all three
Bongard-LOGO families and seals a 6-support/1-query split independently for
each side of every task.  Query identities are therefore fixed before any
support pixel can be released.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


BATCH_SCHEMA = "gkm.bongard-object-batch-plan.v1"
TASK_SCHEMA = "gkm.bongard-object-batch-task.v1"
ALGORITHM_ID = "bongard.object-pipeline/exact-unused-train-stratified-v1"
FAMILIES = ("bd", "ff", "hd")
SIDES = ("side_0", "side_1")
OFFICIAL_SIDE_DIRECTORY = {"side_0": "1", "side_1": "0"}
PANEL_INDEX_DOMAIN = tuple(range(7))

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")


class ObjectBongardBatchError(ValueError):
    """A source commitment, task inventory, or frozen batch is malformed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardBatchError(f"{label} must be a sha256: address")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_selection_or_decision": False,
    }


def object_bongard_batch_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_batch_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-object-batch-algorithm.v1",
            "algorithm_id": ALGORITHM_ID,
            "source_sha256": object_bongard_batch_source_digest(),
            "families": list(FAMILIES),
            "panel_split": "six-support-one-query-per-side",
            "selection": "exact-unused-train-stratified-hash-rank",
            "query_sealed_before_support_pixels": True,
            **_authority_data(),
        }
    )


def _task_id(value: object) -> str:
    if not isinstance(value, str) or _TASK_ID.fullmatch(value) is None:
        raise ObjectBongardBatchError("task ID is outside the official family grammar")
    return value


def _family(task_id: str) -> str:
    family = _task_id(task_id).split("_", 1)[0]
    if family not in FAMILIES:
        raise ObjectBongardBatchError("task family is unsupported")
    return family


def _frozen_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ObjectBongardBatchError(f"{label} must be a task-ID sequence")
    frozen = tuple(values)
    if (
        any(not isinstance(item, str) for item in frozen)
        or frozen != tuple(sorted(set(frozen)))
    ):
        raise ObjectBongardBatchError(f"{label} must be unique and sorted")
    for item in frozen:
        _task_id(item)
    return frozen


def _rank(seed_digest: str, domain: str, value: object) -> str:
    return canonical_digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "selection_seed_digest": seed_digest,
            "domain": domain,
            "value": value,
        }
    )


def object_bongard_task_inventory_digest(task_ids: Sequence[str]) -> str:
    """Digest the official sorted-line task inventory without accepting paths."""

    frozen = _frozen_ids(task_ids, "task inventory")
    payload = "".join(f"{task_id}\n" for task_id in frozen).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _panel_id(family: str, task_id: str, side: str, index: int) -> str:
    if family not in FAMILIES or side not in SIDES or index not in PANEL_INDEX_DOMAIN:
        raise ObjectBongardBatchError("panel role is outside the official 7+7 layout")
    return f"{family}/{task_id}/{OFFICIAL_SIDE_DIRECTORY[side]}/{index}.png"


def _task_content(value: "ObjectBongardTaskPlan") -> dict[str, object]:
    return {
        "schema": TASK_SCHEMA,
        "task_id": value.task_id,
        "family": value.family,
        "split": value.split,
        "side_0_support_panel_ids": list(value.side_0_support_panel_ids),
        "side_1_support_panel_ids": list(value.side_1_support_panel_ids),
        "side_0_query_panel_id": value.side_0_query_panel_id,
        "side_1_query_panel_id": value.side_1_query_panel_id,
        "query_identities_sealed_before_support_pixels": True,
        "support_labels_available_to_python_only": True,
        "query_labels_hidden_until_scoring": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardTaskPlan:
    task_id: str
    family: str
    split: str
    side_0_support_panel_ids: tuple[str, ...]
    side_1_support_panel_ids: tuple[str, ...]
    side_0_query_panel_id: str
    side_1_query_panel_id: str
    record_digest: str

    def __post_init__(self) -> None:
        if _family(self.task_id) != self.family or self.split != "train":
            raise ObjectBongardBatchError("task family or split differs")
        for side, support, query in (
            ("side_0", self.side_0_support_panel_ids, self.side_0_query_panel_id),
            ("side_1", self.side_1_support_panel_ids, self.side_1_query_panel_id),
        ):
            expected = {
                _panel_id(self.family, self.task_id, side, index)
                for index in PANEL_INDEX_DOMAIN
            }
            if (
                not isinstance(support, tuple)
                or len(support) != 6
                or support != tuple(sorted(support))
                or len(set(support)) != 6
                or set(support) | {query} != expected
                or set(support) & {query}
            ):
                raise ObjectBongardBatchError("support/query partition differs")
        _require_address(self.record_digest, "task plan digest")
        if self.record_digest != _address(_task_content(self)):
            raise ObjectBongardBatchError("task plan digest differs")

    @classmethod
    def create(cls, task_id: str, *, seed_digest: str) -> "ObjectBongardTaskPlan":
        task = _task_id(task_id)
        family = _family(task)
        support_by_side: dict[str, tuple[str, ...]] = {}
        query_by_side: dict[str, str] = {}
        for side in SIDES:
            query_index = min(
                PANEL_INDEX_DOMAIN,
                key=lambda index: (
                    _rank(seed_digest, "query-panel", [task, side, index]),
                    index,
                ),
            )
            query_by_side[side] = _panel_id(family, task, side, query_index)
            support_by_side[side] = tuple(
                sorted(
                    _panel_id(family, task, side, index)
                    for index in PANEL_INDEX_DOMAIN
                    if index != query_index
                )
            )
        values: dict[str, object] = {
            "task_id": task,
            "family": family,
            "split": "train",
            "side_0_support_panel_ids": support_by_side["side_0"],
            "side_1_support_panel_ids": support_by_side["side_1"],
            "side_0_query_panel_id": query_by_side["side_0"],
            "side_1_query_panel_id": query_by_side["side_1"],
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_task_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_task_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectBongardTaskPlan":
        expected = {
            "schema",
            "task_id",
            "family",
            "split",
            "side_0_support_panel_ids",
            "side_1_support_panel_ids",
            "side_0_query_panel_id",
            "side_1_query_panel_id",
            "query_identities_sealed_before_support_pixels",
            "support_labels_available_to_python_only",
            "query_labels_hidden_until_scoring",
            *_authority_data(),
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ObjectBongardBatchError("task plan fields differ")
        if (
            value["query_identities_sealed_before_support_pixels"] is not True
            or value["support_labels_available_to_python_only"] is not True
            or value["query_labels_hidden_until_scoring"] is not True
            or any(value[key] != expected_value for key, expected_value in _authority_data().items())
            or not isinstance(value["side_0_support_panel_ids"], list)
            or not isinstance(value["side_1_support_panel_ids"], list)
        ):
            raise ObjectBongardBatchError("task plan policy differs")
        result = cls(
            task_id=value["task_id"],
            family=value["family"],
            split=value["split"],
            side_0_support_panel_ids=tuple(value["side_0_support_panel_ids"]),
            side_1_support_panel_ids=tuple(value["side_1_support_panel_ids"]),
            side_0_query_panel_id=value["side_0_query_panel_id"],
            side_1_query_panel_id=value["side_1_query_panel_id"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise ObjectBongardBatchError("task plan is not canonical")
        return result


def _batch_content(value: "ObjectBongardBatchPlan") -> dict[str, object]:
    return {
        "schema": BATCH_SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_digest": object_bongard_batch_algorithm_digest(),
        "algorithm_source_sha256": object_bongard_batch_source_digest(),
        "selection_seed_digest": value.selection_seed_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "split_source_digest": value.split_source_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "train_task_ids_digest": value.train_task_ids_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "historical_exposure_digest": value.historical_exposure_digest,
        "exact_used_task_ids_digest": value.exact_used_task_ids_digest,
        "candidate_counts": [[family, count] for family, count in value.candidate_counts],
        "requested_per_family": value.requested_per_family,
        "tasks": [item.to_data() for item in value.tasks],
        "selected_task_ids_digest": _address([item.task_id for item in value.tasks]),
        "sealed_query_panel_ids_digest": _address(
            sorted(
                panel_id
                for item in value.tasks
                for panel_id in (item.side_0_query_panel_id, item.side_1_query_panel_id)
            )
        ),
        "selection_inputs_include_pixels": False,
        "selection_inputs_include_action_programs": False,
        "official_test_authorized": False,
        "claim": "exact-unused-train-targeted-engineering-not-official-benchmark",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardBatchPlan:
    selection_seed_digest: str
    release_descriptor_digest: str
    split_source_digest: str
    task_inventory_digest: str
    train_task_ids_digest: str
    exposure_predecessor_digest: str
    historical_exposure_digest: str
    exact_used_task_ids_digest: str
    candidate_counts: tuple[tuple[str, int], ...]
    requested_per_family: int
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
            "historical_exposure_digest",
            "exact_used_task_ids_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            isinstance(self.requested_per_family, bool)
            or not isinstance(self.requested_per_family, int)
            or self.requested_per_family <= 0
            or not isinstance(self.candidate_counts, tuple)
            or len(self.candidate_counts) != len(FAMILIES)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or row[0] != family
                or isinstance(row[1], bool)
                or not isinstance(row[1], int)
                or row[1] < self.requested_per_family
                for family, row in zip(FAMILIES, self.candidate_counts, strict=True)
            )
            or not isinstance(self.tasks, tuple)
            or len(self.tasks) != len(FAMILIES) * self.requested_per_family
            or any(not isinstance(item, ObjectBongardTaskPlan) for item in self.tasks)
            or tuple(item.task_id for item in self.tasks)
            != tuple(sorted(item.task_id for item in self.tasks))
            or tuple(sorted(item.family for item in self.tasks)).count("bd")
            != self.requested_per_family
            or tuple(sorted(item.family for item in self.tasks)).count("ff")
            != self.requested_per_family
            or tuple(sorted(item.family for item in self.tasks)).count("hd")
            != self.requested_per_family
            or self.record_digest != _address(_batch_content(self))
        ):
            raise ObjectBongardBatchError("batch plan identity differs")

    def to_data(self) -> dict[str, object]:
        return {**_batch_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectBongardBatchPlan":
        expected = {
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
            "historical_exposure_digest",
            "exact_used_task_ids_digest",
            "candidate_counts",
            "requested_per_family",
            "tasks",
            "selected_task_ids_digest",
            "sealed_query_panel_ids_digest",
            "selection_inputs_include_pixels",
            "selection_inputs_include_action_programs",
            "official_test_authorized",
            "claim",
            *_authority_data(),
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ObjectBongardBatchError("batch plan fields differ")
        if (
            value["schema"] != BATCH_SCHEMA
            or value["algorithm_id"] != ALGORITHM_ID
            or value["algorithm_digest"] != object_bongard_batch_algorithm_digest()
            or value["algorithm_source_sha256"] != object_bongard_batch_source_digest()
            or value["selection_inputs_include_pixels"] is not False
            or value["selection_inputs_include_action_programs"] is not False
            or value["official_test_authorized"] is not False
            or value["claim"]
            != "exact-unused-train-targeted-engineering-not-official-benchmark"
            or any(value[key] != expected_value for key, expected_value in _authority_data().items())
            or not isinstance(value["candidate_counts"], list)
            or any(
                not isinstance(row, list) or len(row) != 2
                for row in value["candidate_counts"]
            )
            or not isinstance(value["tasks"], list)
            or any(not isinstance(item, Mapping) for item in value["tasks"])
        ):
            raise ObjectBongardBatchError("batch plan policy differs")
        tasks = tuple(ObjectBongardTaskPlan.from_data(item) for item in value["tasks"])
        result = cls(
            selection_seed_digest=value["selection_seed_digest"],
            release_descriptor_digest=value["release_descriptor_digest"],
            split_source_digest=value["split_source_digest"],
            task_inventory_digest=value["task_inventory_digest"],
            train_task_ids_digest=value["train_task_ids_digest"],
            exposure_predecessor_digest=value["exposure_predecessor_digest"],
            historical_exposure_digest=value["historical_exposure_digest"],
            exact_used_task_ids_digest=value["exact_used_task_ids_digest"],
            candidate_counts=tuple(tuple(row) for row in value["candidate_counts"]),
            requested_per_family=value["requested_per_family"],
            tasks=tasks,
            record_digest=value["record_digest"],
        )
        content = _batch_content(result)
        if (
            value["selected_task_ids_digest"] != content["selected_task_ids_digest"]
            or value["sealed_query_panel_ids_digest"]
            != content["sealed_query_panel_ids_digest"]
            or result.to_data() != dict(value)
        ):
            raise ObjectBongardBatchError("batch plan is not canonical")
        return result


def plan_object_bongard_batch(
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    exact_used_task_ids: Sequence[str],
    selection_seed: str,
    requested_per_family: int,
    release_descriptor_digest: str,
    split_source_digest: str,
    task_inventory_digest: str,
    exposure_predecessor_digest: str,
    historical_exposure_digest: str,
) -> ObjectBongardBatchPlan:
    """Select a reproducible cross-family batch without accepting pixel paths."""

    inventory = _frozen_ids(task_ids, "task inventory")
    train = _frozen_ids(train_task_ids, "TRAIN task inventory")
    used = _frozen_ids(exact_used_task_ids, "exact-used task inventory")
    if not set(train) <= set(inventory) or not set(used) <= set(inventory):
        raise ObjectBongardBatchError("TRAIN/used inventory is outside official tasks")
    if (
        not isinstance(selection_seed, str)
        or not selection_seed
        or selection_seed != selection_seed.strip()
        or "\x00" in selection_seed
        or len(selection_seed.encode("utf-8")) > 4096
    ):
        raise ObjectBongardBatchError("selection seed is invalid")
    if (
        isinstance(requested_per_family, bool)
        or not isinstance(requested_per_family, int)
        or requested_per_family <= 0
    ):
        raise ObjectBongardBatchError("requested family count must be positive")
    for name, value in (
        ("release descriptor", release_descriptor_digest),
        ("split source", split_source_digest),
        ("task inventory", task_inventory_digest),
        ("exposure predecessor", exposure_predecessor_digest),
        ("historical exposure", historical_exposure_digest),
    ):
        _require_address(value, name)
    if task_inventory_digest != object_bongard_task_inventory_digest(inventory):
        raise ObjectBongardBatchError("task inventory differs from its source digest")
    seed_digest = "sha256:" + hashlib.sha256(selection_seed.encode("utf-8")).hexdigest()
    unused = set(train) - set(used)
    by_family = {
        family: tuple(sorted(task_id for task_id in unused if _family(task_id) == family))
        for family in FAMILIES
    }
    if any(len(by_family[family]) < requested_per_family for family in FAMILIES):
        raise ObjectBongardBatchError("an official family has too few exact-unused TRAIN tasks")
    selected: list[str] = []
    for family in FAMILIES:
        ranked = sorted(
            by_family[family],
            key=lambda task_id: (
                _rank(seed_digest, f"task:{family}", task_id),
                task_id,
            ),
        )
        selected.extend(ranked[:requested_per_family])
    tasks = tuple(
        sorted(
            (ObjectBongardTaskPlan.create(task_id, seed_digest=seed_digest) for task_id in selected),
            key=lambda item: item.task_id,
        )
    )
    values: dict[str, object] = {
        "selection_seed_digest": seed_digest,
        "release_descriptor_digest": release_descriptor_digest,
        "split_source_digest": split_source_digest,
        "task_inventory_digest": task_inventory_digest,
        "train_task_ids_digest": _address(list(train)),
        "exposure_predecessor_digest": exposure_predecessor_digest,
        "historical_exposure_digest": historical_exposure_digest,
        "exact_used_task_ids_digest": _address(list(used)),
        "candidate_counts": tuple((family, len(by_family[family])) for family in FAMILIES),
        "requested_per_family": requested_per_family,
        "tasks": tasks,
    }
    provisional = object.__new__(ObjectBongardBatchPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardBatchPlan(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_batch_content(provisional)),
    )


def verify_object_bongard_batch_plan(
    plan: ObjectBongardBatchPlan,
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    exact_used_task_ids: Sequence[str],
    selection_seed: str,
) -> ObjectBongardBatchPlan:
    """Cold-reproduce a plan from the same authenticated metadata inputs."""

    if not isinstance(plan, ObjectBongardBatchPlan):
        raise TypeError("plan must be ObjectBongardBatchPlan")
    replay = plan_object_bongard_batch(
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used_task_ids,
        selection_seed=selection_seed,
        requested_per_family=plan.requested_per_family,
        release_descriptor_digest=plan.release_descriptor_digest,
        split_source_digest=plan.split_source_digest,
        task_inventory_digest=plan.task_inventory_digest,
        exposure_predecessor_digest=plan.exposure_predecessor_digest,
        historical_exposure_digest=plan.historical_exposure_digest,
    )
    if replay != plan:
        raise ObjectBongardBatchError("batch plan differs from metadata replay")
    return plan


__all__ = (
    "ALGORITHM_ID",
    "BATCH_SCHEMA",
    "FAMILIES",
    "ObjectBongardBatchError",
    "ObjectBongardBatchPlan",
    "ObjectBongardTaskPlan",
    "object_bongard_batch_algorithm_digest",
    "object_bongard_batch_source_digest",
    "object_bongard_task_inventory_digest",
    "plan_object_bongard_batch",
    "verify_object_bongard_batch_plan",
)
