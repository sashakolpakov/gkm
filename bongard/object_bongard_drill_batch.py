"""Metadata-only planning for the strict ShapeBongard semantic drill.

This module deliberately lives beside, rather than replacing,
``object_bongard_batch``.  The older v1 plan is source-digest bound and remains
the authority for already-preregistered campaigns.  This planner narrows the
new scene-predicate campaign to exact-unused TRAIN tasks whose frozen semantic
cohort is exactly ``drill``.

Selection consumes only authenticated task identifiers, split membership, the
historical exposure seed, and an exact predecessor ledger.  It never accepts a
panel path, opens a PNG, or inspects an action program.  Basic-shape candidates
also pass the conservative morphology policy: numbered variants sharing one
stem are excluded when that stem touches historical exposure or crosses the
drill/dev/sealed partition.  Freeform tasks remain excluded because the frozen
historical seed has no certified unused Freeform-family partition.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.cohorts import BYTE_EXPOSURE_QUALIFICATION, classify_task
from bongard.exposure import (
    ExposureLedger,
    basic_morphology_cluster_id,
    semantic_policy_blocked_keys,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import (
    HistoricalExposureSeed,
    load_historical_exposure,
)
from bongard.object_bongard_batch import (
    ObjectBongardTaskPlan,
    object_bongard_task_inventory_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


BATCH_SCHEMA = "gkm.bongard-object-drill-batch-plan.v1"
ALGORITHM_ID = (
    "bongard.object-pipeline/"
    "exact-unused-train-semantic-drill-bd-hd-disjoint-v1"
)
FAMILIES = ("bd", "hd")
SEMANTIC_COHORT = "drill"
FREEFORM_POLICY = "excluded-no-certified-unused-semantic-partition"
CLAIM = "exact-unused-train-semantic-drill-targeted-engineering-not-official-benchmark"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ObjectBongardDrillBatchError(ValueError):
    """A drill-selection input, policy receipt, or frozen plan is malformed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardDrillBatchError(f"{label} must be a sha256: address")
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


def object_bongard_drill_batch_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_drill_batch_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-object-drill-batch-algorithm.v1",
            "algorithm_id": ALGORITHM_ID,
            "source_sha256": object_bongard_drill_batch_source_digest(),
            "families": list(FAMILIES),
            "semantic_cohort": SEMANTIC_COHORT,
            "freeform_policy": FREEFORM_POLICY,
            "eligibility": (
                "official-TRAIN; exact-task-absent-from-predecessor; "
                "historically-clean; semantic-cohort-drill; "
                "basic-morphology-policy-safe"
            ),
            "selection": (
                "seeded-one-representative-per-generator-cluster; "
                "round-robin-bd-hd; within-batch-disclosure-token-disjoint"
            ),
            "panel_split": "six-support-one-query-per-side",
            "query_sealed_before_support_pixels": True,
            **_authority_data(),
        }
    )


def _seed(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ObjectBongardDrillBatchError("selection seed is invalid")
    return value


def _frozen_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ObjectBongardDrillBatchError(f"{label} must be a task-ID sequence")
    result = tuple(values)
    if (
        any(not isinstance(item, str) or not item for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ObjectBongardDrillBatchError(f"{label} must be unique and sorted")
    return result


def _rank(seed: str, domain: str, value: object) -> str:
    # Keep the already-preregistered public seed's ranking semantics.  The
    # domain and algorithm receipt distinguish this new eligibility policy.
    return canonical_digest(
        {
            "schema": "gkm.bongard-semantic-campaign-ranking.v1",
            "seed": seed,
            "domain": domain,
            "value": value,
        }
    )


@dataclass(frozen=True, slots=True)
class _EligibleTask:
    task_id: str
    family: str
    concepts: tuple[str, ...]

    @property
    def generator_cluster(self) -> tuple[str, ...]:
        if self.family == "bd":
            return tuple(basic_morphology_cluster_id(item) for item in self.concepts)
        return self.concepts

    @property
    def disclosure_tokens(self) -> frozenset[str]:
        if self.family == "bd":
            return frozenset(
                token
                for concept in self.concepts
                for token in (
                    "basic_family:" + concept,
                    "basic_morphology:" + basic_morphology_cluster_id(concept),
                )
            )
        pair = "abstract_pair:" + "\0".join(self.concepts)
        return frozenset(
            {pair}
            | {"abstract_attribute:" + concept for concept in self.concepts}
        )

    def audit_data(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "concepts": list(self.concepts),
            "split": "train",
        }


def _counts(
    values: Sequence[_EligibleTask], families: tuple[str, ...] = FAMILIES
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (family, sum(item.family == family for item in values))
        for family in families
    )


def _select(
    eligible: Sequence[_EligibleTask],
    *,
    selection_seed: str,
    requested_per_family: int,
) -> tuple[_EligibleTask, ...]:
    grouped: dict[str, dict[tuple[str, ...], list[_EligibleTask]]] = {
        family: {} for family in FAMILIES
    }
    for item in eligible:
        grouped[item.family].setdefault(item.generator_cluster, []).append(item)

    queues: dict[str, list[_EligibleTask]] = {}
    for family in FAMILIES:
        representatives = [
            min(
                siblings,
                key=lambda item: (
                    _rank(selection_seed, "within-generator-cluster", item.task_id),
                    item.task_id,
                ),
            )
            for siblings in grouped[family].values()
        ]
        queues[family] = sorted(
            representatives,
            key=lambda item: (
                _rank(
                    selection_seed,
                    "generator-cluster",
                    {"family": item.family, "concepts": list(item.concepts)},
                ),
                item.concepts,
                item.task_id,
            ),
        )

    selected: list[_EligibleTask] = []
    used_tokens: set[str] = set()
    offsets = {family: 0 for family in FAMILIES}
    selected_counts = {family: 0 for family in FAMILIES}
    while any(selected_counts[family] < requested_per_family for family in FAMILIES):
        for family in FAMILIES:
            if selected_counts[family] == requested_per_family:
                continue
            queue = queues[family]
            chosen: _EligibleTask | None = None
            while offsets[family] < len(queue):
                candidate = queue[offsets[family]]
                offsets[family] += 1
                if candidate.disclosure_tokens & used_tokens:
                    continue
                chosen = candidate
                break
            if chosen is None:
                raise ObjectBongardDrillBatchError(
                    f"strict drill policy permits only {selected_counts[family]} "
                    f"disclosure-disjoint {family.upper()} selections"
                )
            selected.append(chosen)
            selected_counts[family] += 1
            used_tokens.update(chosen.disclosure_tokens)
    return tuple(selected)


def _batch_content(value: "ObjectBongardDrillBatchPlan") -> dict[str, object]:
    return {
        "schema": BATCH_SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_digest": object_bongard_drill_batch_algorithm_digest(),
        "algorithm_source_sha256": object_bongard_drill_batch_source_digest(),
        "selection_seed_digest": value.selection_seed_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "split_source_digest": value.split_source_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "train_task_ids_digest": value.train_task_ids_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "historical_exposure_digest": value.historical_exposure_digest,
        "semantic_resolver_policy_digest": value.semantic_resolver_policy_digest,
        "blocked_basic_morphology_clusters_digest": (
            value.blocked_basic_morphology_clusters_digest
        ),
        "clean_cohort_whitelist_digest": value.clean_cohort_whitelist_digest,
        "eligible_task_ids_digest": value.eligible_task_ids_digest,
        "exact_used_task_ids_digest": value.exact_used_task_ids_digest,
        "families": list(FAMILIES),
        "semantic_cohort": SEMANTIC_COHORT,
        "freeform_policy": FREEFORM_POLICY,
        "prepolicy_candidate_counts": [
            [family, count] for family, count in value.prepolicy_candidate_counts
        ],
        "morphology_excluded_counts": [
            [family, count] for family, count in value.morphology_excluded_counts
        ],
        "candidate_counts": [
            [family, count] for family, count in value.candidate_counts
        ],
        "generator_cluster_counts": [
            [family, count] for family, count in value.generator_cluster_counts
        ],
        "requested_per_family": value.requested_per_family,
        "selection_order_task_ids": list(value.selection_order_task_ids),
        "selection_order_task_ids_digest": _address(
            list(value.selection_order_task_ids)
        ),
        "tasks": [item.to_data() for item in value.tasks],
        "selected_task_ids_digest": _address(
            [item.task_id for item in value.tasks]
        ),
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
        "selection_inputs_include_action_programs": False,
        "panel_bytes_opened_during_selection": False,
        "official_test_authorized": False,
        "query_identities_sealed_before_support_pixels": True,
        "qualification": BYTE_EXPOSURE_QUALIFICATION,
        "claim": CLAIM,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardDrillBatchPlan:
    selection_seed_digest: str
    release_descriptor_digest: str
    split_source_digest: str
    task_inventory_digest: str
    train_task_ids_digest: str
    exposure_predecessor_digest: str
    historical_exposure_digest: str
    semantic_resolver_policy_digest: str
    blocked_basic_morphology_clusters_digest: str
    clean_cohort_whitelist_digest: str
    eligible_task_ids_digest: str
    exact_used_task_ids_digest: str
    prepolicy_candidate_counts: tuple[tuple[str, int], ...]
    morphology_excluded_counts: tuple[tuple[str, int], ...]
    candidate_counts: tuple[tuple[str, int], ...]
    generator_cluster_counts: tuple[tuple[str, int], ...]
    requested_per_family: int
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
            "historical_exposure_digest",
            "semantic_resolver_policy_digest",
            "blocked_basic_morphology_clusters_digest",
            "clean_cohort_whitelist_digest",
            "eligible_task_ids_digest",
            "exact_used_task_ids_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        count_fields = (
            self.prepolicy_candidate_counts,
            self.morphology_excluded_counts,
            self.candidate_counts,
            self.generator_cluster_counts,
        )
        historical = load_historical_exposure()
        blocked_clusters = {
            key.concepts[0]
            for key in semantic_policy_blocked_keys(historical)
            if key.kind == "basic_morphology_cluster"
        }
        selected_semantics_are_strict_drill = (
            isinstance(self.tasks, tuple)
            and all(isinstance(task, ObjectBongardTaskPlan) for task in self.tasks)
            and all(
                (
                    (record := classify_task(task.task_id, historical, split=task.split))
                    and task.split == "train"
                    and record.historically_clean
                    and record.semantic_cohort == SEMANTIC_COHORT
                    and (
                        task.family != "bd"
                        or not any(
                            basic_morphology_cluster_id(concept) in blocked_clusters
                            for concept in record.parsed.concepts
                        )
                    )
                )
                for task in self.tasks
            )
        )
        if (
            isinstance(self.requested_per_family, bool)
            or not isinstance(self.requested_per_family, int)
            or self.requested_per_family <= 0
            or any(
                not isinstance(rows, tuple)
                or len(rows) != len(FAMILIES)
                or any(
                    not isinstance(row, tuple)
                    or len(row) != 2
                    or row[0] != family
                    or isinstance(row[1], bool)
                    or not isinstance(row[1], int)
                    or row[1] < 0
                    for family, row in zip(FAMILIES, rows, strict=True)
                )
                for rows in count_fields
            )
            or any(
                before != excluded + after
                for (_, before), (_, excluded), (_, after) in zip(
                    self.prepolicy_candidate_counts,
                    self.morphology_excluded_counts,
                    self.candidate_counts,
                    strict=True,
                )
            )
            or any(count < self.requested_per_family for _, count in self.candidate_counts)
            or any(count < self.requested_per_family for _, count in self.generator_cluster_counts)
            or self.historical_exposure_digest != historical.seed_digest
            or self.semantic_resolver_policy_digest
            != semantic_resolver_policy_digest(historical)
            or self.blocked_basic_morphology_clusters_digest
            != _address(sorted(blocked_clusters))
            or not isinstance(self.tasks, tuple)
            or len(self.tasks) != len(FAMILIES) * self.requested_per_family
            or any(not isinstance(item, ObjectBongardTaskPlan) for item in self.tasks)
            or tuple(item.task_id for item in self.tasks)
            != tuple(sorted(item.task_id for item in self.tasks))
            or any(
                sum(item.family == family for item in self.tasks)
                != self.requested_per_family
                for family in FAMILIES
            )
            or not isinstance(self.selection_order_task_ids, tuple)
            or len(self.selection_order_task_ids) != len(self.tasks)
            or set(self.selection_order_task_ids)
            != {item.task_id for item in self.tasks}
            or any(
                task_id.split("_", 1)[0] != FAMILIES[index % len(FAMILIES)]
                for index, task_id in enumerate(self.selection_order_task_ids)
            )
            or not selected_semantics_are_strict_drill
            or self.record_digest != _address(_batch_content(self))
        ):
            raise ObjectBongardDrillBatchError("drill batch plan identity differs")

    @property
    def algorithm_digest(self) -> str:
        return object_bongard_drill_batch_algorithm_digest()

    @property
    def source_digest(self) -> str:
        return object_bongard_drill_batch_source_digest()

    def to_data(self) -> dict[str, object]:
        return {**_batch_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectBongardDrillBatchPlan":
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
            "semantic_resolver_policy_digest",
            "blocked_basic_morphology_clusters_digest",
            "clean_cohort_whitelist_digest",
            "eligible_task_ids_digest",
            "exact_used_task_ids_digest",
            "families",
            "semantic_cohort",
            "freeform_policy",
            "prepolicy_candidate_counts",
            "morphology_excluded_counts",
            "candidate_counts",
            "generator_cluster_counts",
            "requested_per_family",
            "selection_order_task_ids",
            "selection_order_task_ids_digest",
            "tasks",
            "selected_task_ids_digest",
            "sealed_query_panel_ids_digest",
            "selection_inputs_include_pixels",
            "selection_inputs_include_action_programs",
            "panel_bytes_opened_during_selection",
            "official_test_authorized",
            "query_identities_sealed_before_support_pixels",
            "qualification",
            "claim",
            *_authority_data(),
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ObjectBongardDrillBatchError("drill batch plan fields differ")
        if (
            value["schema"] != BATCH_SCHEMA
            or value["algorithm_id"] != ALGORITHM_ID
            or value["algorithm_digest"]
            != object_bongard_drill_batch_algorithm_digest()
            or value["algorithm_source_sha256"]
            != object_bongard_drill_batch_source_digest()
            or value["families"] != list(FAMILIES)
            or value["semantic_cohort"] != SEMANTIC_COHORT
            or value["freeform_policy"] != FREEFORM_POLICY
            or value["selection_inputs_include_pixels"] is not False
            or value["selection_inputs_include_action_programs"] is not False
            or value["panel_bytes_opened_during_selection"] is not False
            or value["official_test_authorized"] is not False
            or value["query_identities_sealed_before_support_pixels"] is not True
            or value["qualification"] != BYTE_EXPOSURE_QUALIFICATION
            or value["claim"] != CLAIM
            or any(
                value[key] != expected_value
                for key, expected_value in _authority_data().items()
            )
            or any(
                not isinstance(value[name], list)
                for name in (
                    "prepolicy_candidate_counts",
                    "morphology_excluded_counts",
                    "candidate_counts",
                    "generator_cluster_counts",
                    "selection_order_task_ids",
                    "tasks",
                )
            )
        ):
            raise ObjectBongardDrillBatchError("drill batch plan policy differs")
        tasks = tuple(ObjectBongardTaskPlan.from_data(item) for item in value["tasks"])
        result = cls(
            selection_seed_digest=value["selection_seed_digest"],
            release_descriptor_digest=value["release_descriptor_digest"],
            split_source_digest=value["split_source_digest"],
            task_inventory_digest=value["task_inventory_digest"],
            train_task_ids_digest=value["train_task_ids_digest"],
            exposure_predecessor_digest=value["exposure_predecessor_digest"],
            historical_exposure_digest=value["historical_exposure_digest"],
            semantic_resolver_policy_digest=value["semantic_resolver_policy_digest"],
            blocked_basic_morphology_clusters_digest=value[
                "blocked_basic_morphology_clusters_digest"
            ],
            clean_cohort_whitelist_digest=value["clean_cohort_whitelist_digest"],
            eligible_task_ids_digest=value["eligible_task_ids_digest"],
            exact_used_task_ids_digest=value["exact_used_task_ids_digest"],
            prepolicy_candidate_counts=tuple(
                tuple(row) for row in value["prepolicy_candidate_counts"]
            ),
            morphology_excluded_counts=tuple(
                tuple(row) for row in value["morphology_excluded_counts"]
            ),
            candidate_counts=tuple(tuple(row) for row in value["candidate_counts"]),
            generator_cluster_counts=tuple(
                tuple(row) for row in value["generator_cluster_counts"]
            ),
            requested_per_family=value["requested_per_family"],
            selection_order_task_ids=tuple(value["selection_order_task_ids"]),
            tasks=tasks,
            record_digest=value["record_digest"],
        )
        content = _batch_content(result)
        if (
            value["selection_order_task_ids_digest"]
            != content["selection_order_task_ids_digest"]
            or value["selected_task_ids_digest"]
            != content["selected_task_ids_digest"]
            or value["sealed_query_panel_ids_digest"]
            != content["sealed_query_panel_ids_digest"]
            or result.to_data() != dict(value)
        ):
            raise ObjectBongardDrillBatchError("drill batch plan is not canonical")
        return result


def plan_object_bongard_drill_batch(
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    predecessor: ExposureLedger,
    selection_seed: str,
    requested_per_family: int,
    release_descriptor_digest: str,
    split_source_digest: str,
    task_inventory_digest: str,
    historical: HistoricalExposureSeed | None = None,
) -> ObjectBongardDrillBatchPlan:
    """Select a reproducible strict drill batch without accepting visual inputs."""

    if not isinstance(predecessor, ExposureLedger):
        raise TypeError("predecessor must be an ExposureLedger")
    inventory = _frozen_ids(task_ids, "task inventory")
    train = _frozen_ids(train_task_ids, "TRAIN task inventory")
    used = tuple(sorted(predecessor.exposed_task_ids))
    if not set(train) <= set(inventory) or not set(used) <= set(inventory):
        raise ObjectBongardDrillBatchError(
            "TRAIN/predecessor inventory is outside official tasks"
        )
    seed = _seed(selection_seed)
    if (
        isinstance(requested_per_family, bool)
        or not isinstance(requested_per_family, int)
        or requested_per_family <= 0
    ):
        raise ObjectBongardDrillBatchError("requested family count must be positive")
    for name, value in (
        ("release descriptor", release_descriptor_digest),
        ("split source", split_source_digest),
        ("task inventory", task_inventory_digest),
    ):
        _require_address(value, name)
    if task_inventory_digest != object_bongard_task_inventory_digest(inventory):
        raise ObjectBongardDrillBatchError(
            "task inventory differs from its source digest"
        )

    historical_seed = historical or load_historical_exposure()
    resolver_digest = semantic_resolver_policy_digest(historical_seed)
    blocked_clusters = tuple(
        sorted(
            key.concepts[0]
            for key in semantic_policy_blocked_keys(historical_seed)
            if key.kind == "basic_morphology_cluster"
        )
    )
    blocked = set(blocked_clusters)
    prepolicy: list[_EligibleTask] = []
    strict_before_predecessor: list[_EligibleTask] = []
    for task_id in train:
        family = task_id.split("_", 1)[0]
        if family not in FAMILIES:
            continue
        record = classify_task(
            task_id,
            historical_seed,
            split="train",
        )
        if not record.historically_clean or record.semantic_cohort != SEMANTIC_COHORT:
            continue
        item = _EligibleTask(task_id, family, record.parsed.concepts)
        prepolicy.append(item)
        if family == "bd" and any(
            basic_morphology_cluster_id(concept) in blocked
            for concept in item.concepts
        ):
            continue
        strict_before_predecessor.append(item)

    exact_unused_prepolicy = tuple(
        item for item in prepolicy if item.task_id not in predecessor.exposed_task_ids
    )
    eligible = tuple(
        item
        for item in strict_before_predecessor
        if item.task_id not in predecessor.exposed_task_ids
    )
    prepolicy_counts = _counts(exact_unused_prepolicy)
    candidate_counts = _counts(eligible)
    excluded_counts = tuple(
        (
            family,
            dict(prepolicy_counts)[family] - dict(candidate_counts)[family],
        )
        for family in FAMILIES
    )
    cluster_counts = tuple(
        (
            family,
            len(
                {
                    item.generator_cluster
                    for item in eligible
                    if item.family == family
                }
            ),
        )
        for family in FAMILIES
    )
    if any(count < requested_per_family for _, count in candidate_counts):
        raise ObjectBongardDrillBatchError(
            "strict semantic drill has too few exact-unused family candidates"
        )

    selected = _select(
        eligible,
        selection_seed=seed,
        requested_per_family=requested_per_family,
    )
    seed_digest = "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()
    tasks = tuple(
        sorted(
            (
                ObjectBongardTaskPlan.create(
                    item.task_id,
                    seed_digest=seed_digest,
                )
                for item in selected
            ),
            key=lambda item: item.task_id,
        )
    )
    clean_whitelist_data = [
        item.audit_data() for item in sorted(strict_before_predecessor, key=lambda x: x.task_id)
    ]
    eligible_data = [
        item.audit_data() for item in sorted(eligible, key=lambda x: x.task_id)
    ]
    values: dict[str, object] = {
        "selection_seed_digest": seed_digest,
        "release_descriptor_digest": release_descriptor_digest,
        "split_source_digest": split_source_digest,
        "task_inventory_digest": task_inventory_digest,
        "train_task_ids_digest": _address(list(train)),
        "exposure_predecessor_digest": predecessor.digest,
        "historical_exposure_digest": historical_seed.seed_digest,
        "semantic_resolver_policy_digest": resolver_digest,
        "blocked_basic_morphology_clusters_digest": _address(
            list(blocked_clusters)
        ),
        "clean_cohort_whitelist_digest": _address(clean_whitelist_data),
        "eligible_task_ids_digest": _address(eligible_data),
        "exact_used_task_ids_digest": _address(list(used)),
        "prepolicy_candidate_counts": prepolicy_counts,
        "morphology_excluded_counts": excluded_counts,
        "candidate_counts": candidate_counts,
        "generator_cluster_counts": cluster_counts,
        "requested_per_family": requested_per_family,
        "selection_order_task_ids": tuple(item.task_id for item in selected),
        "tasks": tasks,
    }
    provisional = object.__new__(ObjectBongardDrillBatchPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardDrillBatchPlan(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_batch_content(provisional)),
    )


def verify_object_bongard_drill_batch_plan(
    plan: ObjectBongardDrillBatchPlan,
    *,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    predecessor: ExposureLedger,
    selection_seed: str,
    historical: HistoricalExposureSeed | None = None,
) -> ObjectBongardDrillBatchPlan:
    """Cold-reproduce eligibility, selection, and panel partition metadata."""

    if not isinstance(plan, ObjectBongardDrillBatchPlan):
        raise TypeError("plan must be an ObjectBongardDrillBatchPlan")
    replay = plan_object_bongard_drill_batch(
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        predecessor=predecessor,
        selection_seed=selection_seed,
        requested_per_family=plan.requested_per_family,
        release_descriptor_digest=plan.release_descriptor_digest,
        split_source_digest=plan.split_source_digest,
        task_inventory_digest=plan.task_inventory_digest,
        historical=historical,
    )
    if replay != plan:
        raise ObjectBongardDrillBatchError(
            "drill batch plan differs from metadata replay"
        )
    return plan


__all__ = (
    "ALGORITHM_ID",
    "BATCH_SCHEMA",
    "CLAIM",
    "FAMILIES",
    "FREEFORM_POLICY",
    "ObjectBongardDrillBatchError",
    "ObjectBongardDrillBatchPlan",
    "SEMANTIC_COHORT",
    "object_bongard_drill_batch_algorithm_digest",
    "object_bongard_drill_batch_source_digest",
    "plan_object_bongard_drill_batch",
    "verify_object_bongard_drill_batch_plan",
)
