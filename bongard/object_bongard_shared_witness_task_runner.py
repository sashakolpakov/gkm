"""Durable Python runner for structured shared-witness Bongard tasks.

Two ranked full-IR specs are filtered on sealed six-plus-six support evidence.
The first surviving rank, its complete structured predicate, and every entity
observation supporting it are committed and exactly reloaded before a query
callback may expose either query panel.  Query abstentions and errors are
incorrect under the fixed denominator of two; coverage is reported separately.

Cold replay consumes only the archive.  It performs no model calls and has no
Lean, ranker, atlas, polarity-repair, threshold-search, or retry dependency.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
    build_shared_witness_rubric_specs,
)
from bongard.object_bongard_shared_witness_semantics import (
    ObjectBongardSharedWitnessSemanticArtifact,
)
from bongard.object_bongard_shared_witness_slate import (
    ObjectBongardSharedWitnessSlateSelection,
    cold_verify_object_bongard_shared_witness_slate,
    object_bongard_shared_witness_slate_algorithm_digest,
    select_object_bongard_shared_witness_slate,
)
from bongard.object_bongard_shared_witness_support import (
    ObjectBongardSharedWitnessCandidate,
    ObjectBongardSharedWitnessCandidateEvaluation,
    ObjectBongardSharedWitnessSupportVersionSpace,
    SharedWitnessSupportGapKind,
    build_object_bongard_shared_witness_support_version_space,
    cold_verify_object_bongard_shared_witness_support_version_space,
    evaluate_object_bongard_shared_witness_candidate,
)
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID

if TYPE_CHECKING:
    from bongard.object_bongard_shared_witness_observer import (
        ObjectBongardSharedWitnessPanelArtifact,
    )


RUNNER_ID = "bongard.shared-witness-task/two-rank-freeze-query-v1"
QUERY_RESULT_SCHEMA = "gkm.bongard-shared-witness-task-query-result.v1"
FREEZE_SCHEMA = "gkm.bongard-shared-witness-task-freeze.v1"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-shared-witness-task-freeze-commit.v1"
ARCHIVE_SCHEMA = "gkm.bongard-shared-witness-task-run-archive.v1"
SHARED_WITNESS_TASK_SCORE_DENOMINATOR = 2

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardSharedWitnessTaskRunnerError(RuntimeError):
    """A parent, support, freeze, query, or replay boundary failed closed."""


class ObjectBongardSharedWitnessTaskRunStatus(str, Enum):
    COMPLETE = "complete"
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"
    ERROR_GAP = "error_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "lean_removal_changes_decision": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "disjunction_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "threshold_tuning_allowed": False,
        "retries_allowed": False,
        "model_selects_candidate": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            f"{label} fields differ from schema"
        )
    return value


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            f"{label} must be a sha256: address"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessTaskRunnerError(f"{label} is invalid")
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _artifact_type() -> type[ObjectBongardSharedWitnessPanelArtifact]:
    from bongard.object_bongard_shared_witness_observer import (
        ObjectBongardSharedWitnessPanelArtifact,
    )

    return ObjectBongardSharedWitnessPanelArtifact


def object_bongard_shared_witness_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _canonical_parents(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSharedWitnessSemanticArtifact,
    execution_precommit_digest: str,
) -> tuple[
    ObjectBongardTaskPlan,
    ObjectBongardSharedWitnessSemanticArtifact,
    tuple[ObjectBongardSharedWitnessRubricSpec, ObjectBongardSharedWitnessRubricSpec],
    str,
]:
    if not isinstance(task_plan, ObjectBongardTaskPlan):
        raise TypeError("task_plan must be ObjectBongardTaskPlan")
    if not isinstance(semantic_artifact, ObjectBongardSharedWitnessSemanticArtifact):
        raise TypeError("semantic_artifact must be shared-witness semantic evidence")
    plan = ObjectBongardTaskPlan.from_data(task_plan.to_data())
    semantic = ObjectBongardSharedWitnessSemanticArtifact.from_data(
        semantic_artifact.to_data(),
        expected_artifact_digest=semantic_artifact.artifact_digest,
    )
    precommit = _address(execution_precommit_digest, "execution precommit digest")
    if (
        semantic.status is not PrototypeSceneObserverStatus.SUCCESS
        or semantic.task_id != plan.task_id
        or semantic.group_panel_ids
        != (plan.side_0_support_panel_ids, plan.side_1_support_panel_ids)
        or semantic.observation_context_digest != precommit
    ):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "semantic artifact does not bind the exact support plan and precommit"
        )
    specs = build_shared_witness_rubric_specs(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )
    if (
        tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != 2
        or any(item.semantic_artifact_digest != semantic.artifact_digest for item in specs)
    ):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "semantic artifact does not yield the exact two-rank structured slate"
        )
    return plan, semantic, specs, precommit


def _canonical_artifact(
    value: ObjectBongardSharedWitnessPanelArtifact,
) -> ObjectBongardSharedWitnessPanelArtifact:
    artifact_class = _artifact_type()
    if not isinstance(value, artifact_class):
        raise TypeError("evidence must contain shared-witness panel artifacts")
    restored = artifact_class.from_data(value.to_data())
    if restored != value:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "shared-witness observer artifact round trip differs"
        )
    return restored


def _canonical_support(
    plan: ObjectBongardTaskPlan,
    specs: tuple[ObjectBongardSharedWitnessRubricSpec, ObjectBongardSharedWitnessRubricSpec],
    precommit: str,
    side_0: Sequence[Sequence[ObjectBongardSharedWitnessPanelArtifact]],
    side_1: Sequence[Sequence[ObjectBongardSharedWitnessPanelArtifact]],
) -> tuple[
    tuple[tuple[ObjectBongardSharedWitnessPanelArtifact, ...], ...],
    tuple[tuple[ObjectBongardSharedWitnessPanelArtifact, ...], ...],
]:
    if isinstance(side_0, (str, bytes)) or isinstance(side_1, (str, bytes)):
        raise TypeError("ranked support must be a sequence of blocks")
    targets = tuple(
        tuple(sorted((_canonical_artifact(item) for item in block), key=lambda x: x.panel_id))
        for block in side_0
    )
    foils = tuple(
        tuple(sorted((_canonical_artifact(item) for item in block), key=lambda x: x.panel_id))
        for block in side_1
    )
    if len(targets) != 2 or len(foils) != 2:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "support evidence must contain exactly two ranked blocks per side"
        )
    for blocks, expected_ids in (
        (targets, plan.side_0_support_panel_ids),
        (foils, plan.side_1_support_panel_ids),
    ):
        if any(
            len(block) != 6 or tuple(item.panel_id for item in block) != expected_ids
            for block in blocks
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "each rank must bind the exact sorted six-plus-six support IDs"
            )
    artifacts = tuple(
        item for blocks in (targets, foils) for block in blocks for item in block
    )
    if (
        len(artifacts) != 24
        or any(
            item.rubric_spec_digest != specs[rank].spec_digest
            for rank in (0, 1)
            for block in (targets[rank], foils[rank])
            for item in block
        )
        or any(item.observation_context_digest != precommit for item in artifacts)
        or len({item.protocol_digest for item in artifacts}) != 1
        or len({item.runtime_identity_digest for item in artifacts}) != 1
    ):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "ranked support differs in spec, precommit, protocol, or runtime"
        )
    return targets, foils


def _support_digest(
    specs: Sequence[ObjectBongardSharedWitnessRubricSpec],
    targets: Sequence[Sequence[ObjectBongardSharedWitnessPanelArtifact]],
    foils: Sequence[Sequence[ObjectBongardSharedWitnessPanelArtifact]],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-task-support.v1",
            "rubric_specs": [item.to_data() for item in specs],
            "side_0_target_artifacts_by_rank": [
                [item.to_data() for item in block] for block in targets
            ],
            "side_1_foil_artifacts_by_rank": [
                [item.to_data() for item in block] for block in foils
            ],
            "candidate_rank_order": [0, 1],
            "support_observations_per_spec": 12,
            "support_observations_per_task": 24,
            "all_entity_observations_persisted": True,
            "query_material_included": False,
            **_authority_data(),
        }
    )


def _query_result_content(
    value: "ObjectBongardSharedWitnessTaskQueryResult",
) -> dict[str, object]:
    return {
        "schema": QUERY_RESULT_SCHEMA,
        "side": value.side,
        "panel_id": value.panel_id,
        "expected_disposition": value.expected_disposition.value,
        "evaluation": value.evaluation.to_data(),
        "correct": value.correct,
        "incorrect": value.incorrect,
        "covered": value.covered,
        "abstained": value.abstained,
        "fixed_denominator_contribution": 1,
        "indeterminate_or_error_counts_as_incorrect": True,
        "coverage_requires_definite_disposition": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessTaskQueryResult:
    side: str
    panel_id: str
    expected_disposition: Disposition
    evaluation: ObjectBongardSharedWitnessCandidateEvaluation
    correct: bool
    incorrect: bool
    covered: bool
    abstained: bool
    result_digest: str

    def __post_init__(self) -> None:
        expected = (
            Disposition.PRESENT if self.side == "side_0"
            else Disposition.CERTIFIED_ABSENT if self.side == "side_1"
            else None
        )
        if expected is None:
            raise ObjectBongardSharedWitnessTaskRunnerError("query side is unknown")
        if not isinstance(self.evaluation, ObjectBongardSharedWitnessCandidateEvaluation):
            raise TypeError("query evaluation has the wrong type")
        definite = self.evaluation.disposition in (
            Disposition.PRESENT, Disposition.CERTIFIED_ABSENT
        )
        if (
            self.expected_disposition is not expected
            or self.panel_id != self.evaluation.panel_id
            or self.correct is not (self.evaluation.disposition is expected)
            or self.incorrect is not (self.evaluation.disposition is not expected)
            or self.covered is not definite
            or self.abstained is not (not definite)
            or self.result_digest != canonical_digest(_query_result_content(self))
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "query result differs from fixed scoring"
            )

    @classmethod
    def create(
        cls,
        side: str,
        artifact: ObjectBongardSharedWitnessPanelArtifact,
        evaluation: ObjectBongardSharedWitnessCandidateEvaluation,
    ) -> "ObjectBongardSharedWitnessTaskQueryResult":
        expected = (
            Disposition.PRESENT if side == "side_0"
            else Disposition.CERTIFIED_ABSENT if side == "side_1"
            else None
        )
        if expected is None:
            raise ObjectBongardSharedWitnessTaskRunnerError("query side is unknown")
        definite = evaluation.disposition in (
            Disposition.PRESENT, Disposition.CERTIFIED_ABSENT
        )
        values = {
            "side": side,
            "panel_id": artifact.panel_id,
            "expected_disposition": expected,
            "evaluation": evaluation,
            "correct": evaluation.disposition is expected,
            "incorrect": evaluation.disposition is not expected,
            "covered": definite,
            "abstained": not definite,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, result_digest=canonical_digest(_query_result_content(provisional))
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessTaskQueryResult":
        raw = _fields(
            value,
            {
                "schema", "side", "panel_id", "expected_disposition", "evaluation",
                "correct", "incorrect", "covered", "abstained",
                "fixed_denominator_contribution",
                "indeterminate_or_error_counts_as_incorrect",
                "coverage_requires_definite_disposition", *_authority_data(),
                "result_digest",
            },
            "shared-witness query result",
        )
        if (
            raw["schema"] != QUERY_RESULT_SCHEMA
            or raw["fixed_denominator_contribution"] != 1
            or raw["indeterminate_or_error_counts_as_incorrect"] is not True
            or raw["coverage_requires_definite_disposition"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("query result policy differs")
        try:
            expected = Disposition(raw["expected_disposition"])
            result = cls(
                raw["side"], raw["panel_id"], expected,
                ObjectBongardSharedWitnessCandidateEvaluation.from_data(raw["evaluation"]),
                raw["correct"], raw["incorrect"], raw["covered"], raw["abstained"],
                raw["result_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "query result is malformed"
            ) from exc
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "query result is not canonical"
            )
        return result


def _candidate_formula(candidate: ObjectBongardSharedWitnessCandidate) -> str:
    value = candidate.to_data().get("formula")
    if not isinstance(value, str) or value != "shared_witness_has_group_0_axis_endpoint":
        raise ObjectBongardSharedWitnessTaskRunnerError("candidate formula differs")
    return value


def _selected_entity_count(
    space: ObjectBongardSharedWitnessSupportVersionSpace,
) -> int:
    return sum(len(item.observation.entities) for item in space.support_artifacts)


def _freeze_content(value: "ObjectBongardSharedWitnessTaskFreeze") -> dict[str, object]:
    selection = value.slate_selection
    selected = selection.selected_candidate
    selected_spec = selection.selected_rubric_spec
    selected_space = selection.selected_version_space
    if selected is None or selected_spec is None or selected_space is None:
        raise ObjectBongardSharedWitnessTaskRunnerError("cannot serialize an empty freeze")
    return {
        "schema": FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "rubric_spec_digests": [item.spec_digest for item in value.rubric_specs],
        "support_digest": value.support_digest,
        "observer_protocol_digest": value.observer_protocol_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "slate_algorithm_digest": value.slate_algorithm_digest,
        "version_space_digests": list(value.version_space_digests),
        "slate_selection": selection.to_data(),
        "slate_selection_digest": selection.selection_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_response_digest_is_legacy_alias_for_slate_selection": True,
        "selected_rubric_spec": selected_spec.to_data(),
        "selected_rubric_spec_digest": selected_spec.spec_digest,
        "selected_candidate": selected.to_data(),
        "selected_candidate_digest": selected.candidate_digest,
        "selected_candidate_rank": selected.candidate_rank,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_formula": value.selected_formula,
        "selected_support_version_space": selected_space.to_data(),
        "selected_support_version_space_digest": selected_space.version_space_digest,
        "selected_support_artifact_count": len(selected_space.support_artifacts),
        "selected_entity_evidence_count": _selected_entity_count(selected_space),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_bytes_included": False,
        "query_observer_artifacts_included": False,
        "full_structured_ir_frozen_before_query_source": True,
        "selected_rank_formula_and_all_support_entities_frozen": True,
        "candidate_order": ["rank-0/group-0-target", "rank-1/group-0-target"],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessTaskFreeze:
    """Full selected IR and entity evidence committed before query access."""

    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[
        ObjectBongardSharedWitnessRubricSpec,
        ObjectBongardSharedWitnessRubricSpec,
    ]
    support_digest: str
    observer_protocol_digest: str
    observer_runtime_identity_digest: str
    slate_algorithm_digest: str
    version_space_digests: tuple[str, str]
    slate_selection: ObjectBongardSharedWitnessSlateSelection
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    selected_formula: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.runner_source_digest, "runner source digest")
        _identifier(self.task_id, "task ID")
        _address(self.task_plan_digest, "task plan digest")
        _address(self.execution_precommit_digest, "execution precommit digest")
        for name in (
            "semantic_artifact_digest", "support_digest", "observer_protocol_digest",
            "observer_runtime_identity_digest", "slate_algorithm_digest",
            "version_space_digest", "support_version_space_digest",
            "rank_response_digest", "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.record_digest, "freeze record digest")
        if (
            not isinstance(self.rubric_specs, tuple)
            or len(self.rubric_specs) != 2
            or any(not isinstance(item, ObjectBongardSharedWitnessRubricSpec) for item in self.rubric_specs)
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze requires two full-IR specs")
        specs = tuple(
            ObjectBongardSharedWitnessRubricSpec.from_data(item.to_data())
            for item in self.rubric_specs
        )
        if not isinstance(self.slate_selection, ObjectBongardSharedWitnessSlateSelection):
            raise TypeError("freeze slate selection has the wrong type")
        selection = ObjectBongardSharedWitnessSlateSelection.from_data(
            self.slate_selection.to_data()
        )
        selected = selection.selected_candidate
        selected_spec = selection.selected_rubric_spec
        selected_space = selection.selected_version_space
        if (
            selected is None or selected_spec is None or selected_space is None
            or specs != self.rubric_specs
            or tuple(item.candidate_rank for item in specs) != (0, 1)
            or self.runner_source_digest != object_bongard_shared_witness_task_runner_source_digest()
            or selection != self.slate_selection
            or selection.rubric_specs != specs
            or self.semantic_artifact_digest != selection.semantic_artifact_digest
            or self.slate_algorithm_digest != object_bongard_shared_witness_slate_algorithm_digest()
            or self.slate_algorithm_digest != selection.algorithm_digest
            or self.version_space_digests != tuple(item.version_space_digest for item in selection.version_spaces)
            or self.observer_protocol_digest != selected_space.observer_protocol_digest
            or self.observer_runtime_identity_digest != selected_space.observer_runtime_identity_digest
            or len({self.version_space_digest, self.support_version_space_digest,
                    self.rank_response_digest, selection.selection_digest}) != 1
            or self.selected_predicate_digest != selected.candidate_digest
            or self.selected_formula != _candidate_formula(selected)
            or self.sealed_query_panel_ids != tuple(self.sealed_query_panel_ids)
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(_IDENTIFIER.fullmatch(item) is None for item in self.sealed_query_panel_ids)
            or len(selected_space.support_artifacts) != 12
            or self.record_digest != _content_address(_freeze_content(self))
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("task freeze content differs")

    @property
    def selected_candidate(self) -> ObjectBongardSharedWitnessCandidate:
        selected = self.slate_selection.selected_candidate
        if selected is None:
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze selection is empty")
        return selected

    @property
    def selected_rubric_spec(self) -> ObjectBongardSharedWitnessRubricSpec:
        selected = self.slate_selection.selected_rubric_spec
        if selected is None:
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze selection is empty")
        return selected

    @property
    def selected_support_version_space(self) -> ObjectBongardSharedWitnessSupportVersionSpace:
        selected = self.slate_selection.selected_version_space
        if selected is None:
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze selection is empty")
        return selected

    @property
    def rubric_spec_digest(self) -> str:
        return self.selected_rubric_spec.spec_digest

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        semantic_artifact: ObjectBongardSharedWitnessSemanticArtifact,
        rubric_specs: tuple[
            ObjectBongardSharedWitnessRubricSpec,
            ObjectBongardSharedWitnessRubricSpec,
        ],
        support_digest: str,
        slate_selection: ObjectBongardSharedWitnessSlateSelection,
    ) -> "ObjectBongardSharedWitnessTaskFreeze":
        selected = slate_selection.selected_candidate
        selected_space = slate_selection.selected_version_space
        if selected is None or selected_space is None:
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "cannot freeze a slate without a support survivor"
            )
        values = {
            "runner_source_digest": object_bongard_shared_witness_task_runner_source_digest(),
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.record_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "semantic_artifact_digest": semantic_artifact.artifact_digest,
            "rubric_specs": rubric_specs,
            "support_digest": support_digest,
            "observer_protocol_digest": selected_space.observer_protocol_digest,
            "observer_runtime_identity_digest": selected_space.observer_runtime_identity_digest,
            "slate_algorithm_digest": slate_selection.algorithm_digest,
            "version_space_digests": tuple(
                item.version_space_digest for item in slate_selection.version_spaces
            ),
            "slate_selection": slate_selection,
            "version_space_digest": slate_selection.selection_digest,
            "support_version_space_digest": slate_selection.selection_digest,
            "rank_response_digest": slate_selection.selection_digest,
            "selected_predicate_digest": selected.candidate_digest,
            "selected_formula": _candidate_formula(selected),
            "sealed_query_panel_ids": (
                task_plan.side_0_query_panel_id, task_plan.side_1_query_panel_id
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_content_address(_freeze_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest", "semantic_artifact_digest",
                "rubric_specs", "rubric_spec_digests", "support_digest",
                "observer_protocol_digest", "observer_runtime_identity_digest",
                "slate_algorithm_digest", "version_space_digests", "slate_selection",
                "slate_selection_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_rubric_spec", "selected_rubric_spec_digest",
                "selected_candidate", "selected_candidate_digest", "selected_candidate_rank",
                "selected_predicate_digest", "selected_formula",
                "selected_support_version_space", "selected_support_version_space_digest",
                "selected_support_artifact_count", "selected_entity_evidence_count",
                "sealed_query_panel_ids", "query_bytes_included",
                "query_observer_artifacts_included",
                "full_structured_ir_frozen_before_query_source",
                "selected_rank_formula_and_all_support_entities_frozen",
                "candidate_order", *_authority_data(), "record_digest",
            },
            "shared-witness task freeze",
        )
        for name in ("rubric_specs", "rubric_spec_digests", "version_space_digests", "sealed_query_panel_ids", "candidate_order"):
            if not isinstance(raw[name], list):
                raise ObjectBongardSharedWitnessTaskRunnerError(f"freeze {name} must be a list")
        specs = tuple(
            ObjectBongardSharedWitnessRubricSpec.from_data(item)
            for item in raw["rubric_specs"]
        )
        selection = ObjectBongardSharedWitnessSlateSelection.from_data(raw["slate_selection"])
        selected = selection.selected_candidate
        selected_spec = selection.selected_rubric_spec
        selected_space = selection.selected_version_space
        if (
            raw["schema"] != FREEZE_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["slate_selection_digest"] != selection.selection_digest
            or selected is None or selected_spec is None or selected_space is None
            or raw["selected_rubric_spec"] != selected_spec.to_data()
            or raw["selected_rubric_spec_digest"] != selected_spec.spec_digest
            or raw["selected_candidate"] != selected.to_data()
            or raw["selected_candidate_digest"] != selected.candidate_digest
            or raw["selected_candidate_rank"] != selected.candidate_rank
            or raw["selected_support_version_space"] != selected_space.to_data()
            or raw["selected_support_version_space_digest"] != selected_space.version_space_digest
            or raw["selected_support_artifact_count"] != 12
            or raw["selected_entity_evidence_count"] != _selected_entity_count(selected_space)
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["query_bytes_included"] is not False
            or raw["query_observer_artifacts_included"] is not False
            or raw["full_structured_ir_frozen_before_query_source"] is not True
            or raw["selected_rank_formula_and_all_support_entities_frozen"] is not True
            or raw["candidate_order"] != ["rank-0/group-0-target", "rank-1/group-0-target"]
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze policy differs")
        result = cls(
            raw["runner_source_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["semantic_artifact_digest"], specs,
            raw["support_digest"], raw["observer_protocol_digest"],
            raw["observer_runtime_identity_digest"], raw["slate_algorithm_digest"],
            tuple(raw["version_space_digests"]), selection, raw["version_space_digest"],
            raw["support_version_space_digest"], raw["rank_response_digest"],
            raw["selected_predicate_digest"], raw["selected_formula"],
            tuple(raw["sealed_query_panel_ids"]), raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze is not canonical")
        return result


def _commit_content(value: "ObjectBongardSharedWitnessTaskFreezeCommit") -> dict[str, object]:
    return {
        "schema": FREEZE_COMMIT_SCHEMA,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "slate_selection_digest": value.slate_selection_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_response_digest_is_legacy_alias_for_slate_selection": True,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "durably_persisted_and_reloaded_before_query_bytes": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessTaskFreezeCommit:
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    slate_selection_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.task_id, "commit task ID")
        _address(self.task_plan_digest, "commit task plan digest")
        _address(self.execution_precommit_digest, "commit precommit digest")
        for name in (
            "slate_selection_digest", "version_space_digest",
            "support_version_space_digest", "rank_response_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        for name in (
            "task_freeze_digest", "exact_freeze_payload_digest",
            "task_freeze_store_receipt_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        if (
            len({self.slate_selection_digest, self.version_space_digest,
                 self.support_version_space_digest, self.rank_response_digest}) != 1
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze commit differs")

    @classmethod
    def seal(
        cls,
        freeze: ObjectBongardSharedWitnessTaskFreeze,
        exact_freeze_payload: bytes,
        *,
        task_freeze_store_receipt_digest: str,
    ) -> "ObjectBongardSharedWitnessTaskFreezeCommit":
        if not isinstance(freeze, ObjectBongardSharedWitnessTaskFreeze):
            raise TypeError("freeze must be ObjectBongardSharedWitnessTaskFreeze")
        expected = canonical_json(freeze.to_data()) + b"\n"
        if exact_freeze_payload != expected:
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "freeze payload is not exact canonical JSON"
            )
        values = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "slate_selection_digest": freeze.slate_selection.selection_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "rank_response_digest": freeze.rank_response_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": "sha256:" + hashlib.sha256(expected).hexdigest(),
            "task_freeze_store_receipt_digest": _address(
                task_freeze_store_receipt_digest, "freeze store receipt digest"
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_content_address(_commit_content(provisional)))

    def assert_matches(
        self,
        freeze: ObjectBongardSharedWitnessTaskFreeze,
        exact_freeze_payload: bytes,
    ) -> None:
        if self != type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=self.task_freeze_store_receipt_digest,
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze commit replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "task_id", "task_plan_digest", "execution_precommit_digest",
                "slate_selection_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "task_freeze_store_receipt_digest",
                "durably_persisted_and_reloaded_before_query_bytes",
                *_authority_data(), "record_digest",
            },
            "shared-witness freeze commit",
        )
        if (
            raw["schema"] != FREEZE_COMMIT_SCHEMA
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["durably_persisted_and_reloaded_before_query_bytes"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze commit policy differs")
        result = cls(
            raw["task_id"], raw["task_plan_digest"], raw["execution_precommit_digest"],
            raw["slate_selection_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["rank_response_digest"],
            raw["selected_predicate_digest"], raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"], raw["task_freeze_store_receipt_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessTaskRunnerError("freeze commit is not canonical")
        return result


FreezeCommitter = Callable[
    [bytes], ObjectBongardSharedWitnessTaskFreezeCommit | Mapping[str, Any]
]
FreezeReloader = Callable[[Mapping[str, Any]], bytes]
QuerySource = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, "ObjectBongardSharedWitnessPanelArtifact"],
]


def _gap_status(
    spaces: Sequence[ObjectBongardSharedWitnessSupportVersionSpace],
) -> ObjectBongardSharedWitnessTaskRunStatus:
    if any(item.survivor_candidate_digests for item in spaces):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "empty slate cannot contain a support survivor"
        )
    gaps = tuple(item.gap for item in spaces)
    if any(item is None for item in gaps):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "empty slate lacks a typed support gap"
        )
    kinds = {item.kind for item in gaps if item is not None}
    if SharedWitnessSupportGapKind.ERROR_GAP in kinds:
        return ObjectBongardSharedWitnessTaskRunStatus.ERROR_GAP
    if SharedWitnessSupportGapKind.WITNESS_GAP in kinds:
        return ObjectBongardSharedWitnessTaskRunStatus.WITNESS_GAP
    return ObjectBongardSharedWitnessTaskRunStatus.LANGUAGE_GAP


def _archive_content(
    value: "ObjectBongardSharedWitnessTaskRunArchive",
) -> dict[str, object]:
    return {
        "schema": ARCHIVE_SCHEMA,
        "runner_id": RUNNER_ID,
        "status": value.status.value,
        "runner_source_digest": value.runner_source_digest,
        "task_plan": value.task_plan.to_data(),
        "task_plan_digest": value.task_plan.record_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "semantic_artifact": value.semantic_artifact.to_data(),
        "semantic_artifact_digest": value.semantic_artifact.artifact_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "rubric_spec_digests": [item.spec_digest for item in value.rubric_specs],
        "side_0_support_by_rank": [
            [item.to_data() for item in block] for block in value.side_0_support_by_rank
        ],
        "side_1_support_by_rank": [
            [item.to_data() for item in block] for block in value.side_1_support_by_rank
        ],
        "support_digest": value.support_digest,
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "version_space_digests": [item.version_space_digest for item in value.version_spaces],
        "slate_selection": value.slate_selection.to_data(),
        "slate_selection_digest": value.slate_selection.selection_digest,
        "selection_model_calls_made": 0,
        "selection_replay_calls_made": value.selection_replay_calls_made,
        "freeze": None if value.freeze is None else value.freeze.to_data(),
        "freeze_commit": None if value.freeze_commit is None else value.freeze_commit.to_data(),
        "side_0_query": None if value.side_0_query is None else value.side_0_query.to_data(),
        "side_1_query": None if value.side_1_query is None else value.side_1_query.to_data(),
        "query_results": [item.to_data() for item in value.query_results],
        "correct_count": value.correct_count,
        "incorrect_count": value.incorrect_count,
        "abstention_count": value.abstention_count,
        "coverage_count": value.coverage_count,
        "score_denominator": value.score_denominator,
        "accuracy_ppm": value.accuracy_ppm,
        "coverage_ppm": value.coverage_ppm,
        "freeze_commit_calls_made": value.freeze_commit_calls_made,
        "freeze_reload_calls_made": value.freeze_reload_calls_made,
        "query_source_calls_made": value.query_source_calls_made,
        "support_spec_count": 2,
        "support_observations_per_spec": 12,
        "support_observations_per_task": 24,
        "query_observations_per_task": 2,
        "fixed_query_denominator": 2,
        "candidate_order": ["rank-0/group-0-target", "rank-1/group-0-target"],
        "full_structured_ir_and_all_entity_observations_persisted": True,
        "selected_rank_formula_frozen_before_query_bytes": True,
        "query_source_called_only_after_exact_freeze_reload": True,
        "gap_counts_as_two_uncovered_incorrect_abstentions": True,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessTaskRunArchive:
    """Self-contained support, freeze, query, score, and replay record."""

    status: ObjectBongardSharedWitnessTaskRunStatus
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    semantic_artifact: ObjectBongardSharedWitnessSemanticArtifact
    rubric_specs: tuple[
        ObjectBongardSharedWitnessRubricSpec,
        ObjectBongardSharedWitnessRubricSpec,
    ]
    side_0_support_by_rank: tuple[
        tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
        tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
    ]
    side_1_support_by_rank: tuple[
        tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
        tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
    ]
    support_digest: str
    version_spaces: tuple[
        ObjectBongardSharedWitnessSupportVersionSpace,
        ObjectBongardSharedWitnessSupportVersionSpace,
    ]
    slate_selection: ObjectBongardSharedWitnessSlateSelection
    freeze: ObjectBongardSharedWitnessTaskFreeze | None
    freeze_commit: ObjectBongardSharedWitnessTaskFreezeCommit | None
    side_0_query: ObjectBongardSharedWitnessPanelArtifact | None
    side_1_query: ObjectBongardSharedWitnessPanelArtifact | None
    query_results: tuple[ObjectBongardSharedWitnessTaskQueryResult, ...]
    correct_count: int
    incorrect_count: int
    abstention_count: int
    coverage_count: int
    score_denominator: int
    accuracy_ppm: int
    coverage_ppm: int
    selection_replay_calls_made: int
    freeze_commit_calls_made: int
    freeze_reload_calls_made: int
    query_source_calls_made: int
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObjectBongardSharedWitnessTaskRunStatus):
            raise TypeError("task status has the wrong type")
        if self.runner_source_digest != object_bongard_shared_witness_task_runner_source_digest():
            raise ObjectBongardSharedWitnessTaskRunnerError("runner source binding differs")
        _raw_digest(self.record_digest, "archive digest")
        plan, semantic, specs, precommit = _canonical_parents(
            self.task_plan, self.semantic_artifact, self.execution_precommit_digest
        )
        if (
            plan != self.task_plan or semantic != self.semantic_artifact
            or specs != self.rubric_specs or precommit != self.execution_precommit_digest
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("archive parent binding differs")
        targets, foils = _canonical_support(
            plan, specs, precommit,
            self.side_0_support_by_rank, self.side_1_support_by_rank,
        )
        if targets != self.side_0_support_by_rank or foils != self.side_1_support_by_rank:
            raise ObjectBongardSharedWitnessTaskRunnerError("archive support differs")
        expected_support_digest = _support_digest(specs, targets, foils)
        rebuilt_spaces = tuple(
            cold_verify_object_bongard_shared_witness_support_version_space(
                build_object_bongard_shared_witness_support_version_space(
                    specs[rank], targets[rank], foils[rank]
                ),
                specs[rank], targets[rank], foils[rank],
            )
            for rank in (0, 1)
        )
        rebuilt_selection = cold_verify_object_bongard_shared_witness_slate(
            select_object_bongard_shared_witness_slate(specs, rebuilt_spaces),
            specs,
            rebuilt_spaces,
        )
        if (
            self.support_digest != expected_support_digest
            or self.version_spaces != rebuilt_spaces
            or self.slate_selection != rebuilt_selection
            or self.selection_replay_calls_made != 1
            or self.score_denominator != SHARED_WITNESS_TASK_SCORE_DENOMINATOR
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError(
                "archive support selection replay differs"
            )
        selected = rebuilt_selection.selected_candidate
        if selected is None:
            expected_status = _gap_status(rebuilt_spaces)
            if (
                self.status is not expected_status
                or self.freeze is not None or self.freeze_commit is not None
                or self.side_0_query is not None or self.side_1_query is not None
                or self.query_results != ()
                or (self.correct_count, self.incorrect_count, self.abstention_count,
                    self.coverage_count, self.accuracy_ppm, self.coverage_ppm)
                != (0, 2, 2, 0, 0, 0)
                or (self.freeze_commit_calls_made, self.freeze_reload_calls_made,
                    self.query_source_calls_made) != (0, 0, 0)
            ):
                raise ObjectBongardSharedWitnessTaskRunnerError("gap archive differs")
        else:
            if self.status is not ObjectBongardSharedWitnessTaskRunStatus.COMPLETE:
                raise ObjectBongardSharedWitnessTaskRunnerError("selected archive is not complete")
            if not isinstance(self.freeze, ObjectBongardSharedWitnessTaskFreeze) or not isinstance(
                self.freeze_commit, ObjectBongardSharedWitnessTaskFreezeCommit
            ):
                raise ObjectBongardSharedWitnessTaskRunnerError("complete archive lacks freeze custody")
            expected_freeze = ObjectBongardSharedWitnessTaskFreeze.seal(
                task_plan=plan,
                execution_precommit_digest=precommit,
                semantic_artifact=semantic,
                rubric_specs=specs,
                support_digest=expected_support_digest,
                slate_selection=rebuilt_selection,
            )
            freeze_bytes = canonical_json(expected_freeze.to_data()) + b"\n"
            self.freeze_commit.assert_matches(expected_freeze, freeze_bytes)
            if self.freeze != expected_freeze:
                raise ObjectBongardSharedWitnessTaskRunnerError("archive freeze replay differs")
            if self.side_0_query is None or self.side_1_query is None:
                raise ObjectBongardSharedWitnessTaskRunnerError("complete archive lacks query evidence")
            queries = (
                _canonical_artifact(self.side_0_query),
                _canonical_artifact(self.side_1_query),
            )
            selected_spec = rebuilt_selection.selected_rubric_spec
            selected_space = rebuilt_selection.selected_version_space
            if selected_spec is None or selected_space is None:
                raise ObjectBongardSharedWitnessTaskRunnerError("selected slate lacks bindings")
            if (
                tuple(item.panel_id for item in queries)
                != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
                or any(item.rubric_spec_digest != selected_spec.spec_digest for item in queries)
                or any(item.observation_context_digest != precommit for item in queries)
                or any(item.protocol_digest != selected_space.observer_protocol_digest for item in queries)
                or any(item.runtime_identity_digest != selected_space.observer_runtime_identity_digest for item in queries)
            ):
                raise ObjectBongardSharedWitnessTaskRunnerError(
                    "query artifacts differ from frozen spec or runtime"
                )
            evaluations = tuple(
                evaluate_object_bongard_shared_witness_candidate(selected, item)
                for item in queries
            )
            expected_results = tuple(
                ObjectBongardSharedWitnessTaskQueryResult.create(side, artifact, evaluation)
                for side, artifact, evaluation in zip(
                    ("side_0", "side_1"), queries, evaluations, strict=True
                )
            )
            correct = sum(item.correct for item in expected_results)
            incorrect = sum(item.incorrect for item in expected_results)
            abstained = sum(item.abstained for item in expected_results)
            covered = sum(item.covered for item in expected_results)
            if (
                self.query_results != expected_results
                or self.correct_count != correct or self.incorrect_count != incorrect
                or self.abstention_count != abstained or self.coverage_count != covered
                or self.accuracy_ppm != correct * 500_000
                or self.coverage_ppm != covered * 500_000
                or (self.freeze_commit_calls_made, self.freeze_reload_calls_made,
                    self.query_source_calls_made) != (1, 1, 1)
            ):
                raise ObjectBongardSharedWitnessTaskRunnerError("query score replay differs")
        if self.record_digest != canonical_digest(_archive_content(self)):
            raise ObjectBongardSharedWitnessTaskRunnerError("archive digest differs")

    @property
    def selected_candidate(self) -> ObjectBongardSharedWitnessCandidate | None:
        return self.slate_selection.selected_candidate

    @property
    def selected_rubric_spec(self) -> ObjectBongardSharedWitnessRubricSpec | None:
        return self.slate_selection.selected_rubric_spec

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "status", "runner_source_digest", "task_plan",
                "task_plan_digest", "execution_precommit_digest", "semantic_artifact",
                "semantic_artifact_digest", "rubric_specs", "rubric_spec_digests",
                "side_0_support_by_rank", "side_1_support_by_rank", "support_digest",
                "version_spaces", "version_space_digests", "slate_selection",
                "slate_selection_digest", "selection_model_calls_made",
                "selection_replay_calls_made", "freeze", "freeze_commit",
                "side_0_query", "side_1_query", "query_results", "correct_count",
                "incorrect_count", "abstention_count", "coverage_count",
                "score_denominator", "accuracy_ppm", "coverage_ppm",
                "freeze_commit_calls_made", "freeze_reload_calls_made",
                "query_source_calls_made", "support_spec_count",
                "support_observations_per_spec", "support_observations_per_task",
                "query_observations_per_task", "fixed_query_denominator",
                "candidate_order", "full_structured_ir_and_all_entity_observations_persisted",
                "selected_rank_formula_frozen_before_query_bytes",
                "query_source_called_only_after_exact_freeze_reload",
                "gap_counts_as_two_uncovered_incorrect_abstentions",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "shared-witness task archive",
        )
        list_fields = (
            "rubric_specs", "rubric_spec_digests", "side_0_support_by_rank",
            "side_1_support_by_rank", "version_spaces", "version_space_digests",
            "query_results", "candidate_order",
        )
        if any(not isinstance(raw[name], list) for name in list_fields):
            raise ObjectBongardSharedWitnessTaskRunnerError("archive list field differs")
        if (
            raw["schema"] != ARCHIVE_SCHEMA or raw["runner_id"] != RUNNER_ID
            or raw["selection_model_calls_made"] != 0
            or raw["support_spec_count"] != 2
            or raw["support_observations_per_spec"] != 12
            or raw["support_observations_per_task"] != 24
            or raw["query_observations_per_task"] != 2
            or raw["fixed_query_denominator"] != 2
            or raw["candidate_order"] != ["rank-0/group-0-target", "rank-1/group-0-target"]
            or raw["full_structured_ir_and_all_entity_observations_persisted"] is not True
            or raw["selected_rank_formula_frozen_before_query_bytes"] is not True
            or raw["query_source_called_only_after_exact_freeze_reload"] is not True
            or raw["gap_counts_as_two_uncovered_incorrect_abstentions"] is not True
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("archive policy differs")
        artifact_class = _artifact_type()
        try:
            plan = ObjectBongardTaskPlan.from_data(raw["task_plan"])
            semantic = ObjectBongardSharedWitnessSemanticArtifact.from_data(
                raw["semantic_artifact"],
                expected_artifact_digest=raw["semantic_artifact_digest"],
            )
            specs = tuple(
                ObjectBongardSharedWitnessRubricSpec.from_data(item)
                for item in raw["rubric_specs"]
            )
            targets = tuple(
                tuple(artifact_class.from_data(item) for item in block)
                for block in raw["side_0_support_by_rank"]
            )
            foils = tuple(
                tuple(artifact_class.from_data(item) for item in block)
                for block in raw["side_1_support_by_rank"]
            )
            spaces = tuple(
                ObjectBongardSharedWitnessSupportVersionSpace.from_data(item)
                for item in raw["version_spaces"]
            )
            selection = ObjectBongardSharedWitnessSlateSelection.from_data(raw["slate_selection"])
            status = ObjectBongardSharedWitnessTaskRunStatus(raw["status"])
            result = cls(
                status, raw["runner_source_digest"], plan,
                raw["execution_precommit_digest"], semantic, specs, targets, foils,
                raw["support_digest"], spaces, selection,
                None if raw["freeze"] is None else ObjectBongardSharedWitnessTaskFreeze.from_data(raw["freeze"]),
                None if raw["freeze_commit"] is None else ObjectBongardSharedWitnessTaskFreezeCommit.from_data(raw["freeze_commit"]),
                None if raw["side_0_query"] is None else artifact_class.from_data(raw["side_0_query"]),
                None if raw["side_1_query"] is None else artifact_class.from_data(raw["side_1_query"]),
                tuple(ObjectBongardSharedWitnessTaskQueryResult.from_data(item) for item in raw["query_results"]),
                raw["correct_count"], raw["incorrect_count"], raw["abstention_count"],
                raw["coverage_count"], raw["score_denominator"], raw["accuracy_ppm"],
                raw["coverage_ppm"], raw["selection_replay_calls_made"],
                raw["freeze_commit_calls_made"], raw["freeze_reload_calls_made"],
                raw["query_source_calls_made"], raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessTaskRunnerError("archive is malformed") from exc
        if (
            raw["task_plan_digest"] != plan.record_digest
            or raw["semantic_artifact_digest"] != semantic.artifact_digest
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["version_space_digests"] != [item.version_space_digest for item in spaces]
            or raw["slate_selection_digest"] != selection.selection_digest
            or result.to_data() != dict(raw)
        ):
            raise ObjectBongardSharedWitnessTaskRunnerError("archive is not canonical")
        return result


def _make_archive(**values: object) -> ObjectBongardSharedWitnessTaskRunArchive:
    provisional = object.__new__(ObjectBongardSharedWitnessTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessTaskRunArchive(
        **values, record_digest=canonical_digest(_archive_content(provisional))
    )


def run_object_bongard_shared_witness_task(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSharedWitnessSemanticArtifact,
    side_0_support_by_rank: Sequence[
        Sequence[ObjectBongardSharedWitnessPanelArtifact]
    ],
    side_1_support_by_rank: Sequence[
        Sequence[ObjectBongardSharedWitnessPanelArtifact]
    ],
    *,
    execution_precommit_digest: str,
    freeze_committer: FreezeCommitter,
    freeze_reloader: FreezeReloader,
    query_source: QuerySource,
) -> ObjectBongardSharedWitnessTaskRunArchive:
    """Select from 24 support calls, freeze, then request exactly two queries."""

    plan, semantic, specs, precommit = _canonical_parents(
        task_plan, semantic_artifact, execution_precommit_digest
    )
    targets, foils = _canonical_support(
        plan, specs, precommit, side_0_support_by_rank, side_1_support_by_rank
    )
    spaces = tuple(
        cold_verify_object_bongard_shared_witness_support_version_space(
            built, specs[rank], targets[rank], foils[rank]
        )
        for rank in (0, 1)
        for built in (
            build_object_bongard_shared_witness_support_version_space(
                specs[rank], targets[rank], foils[rank]
            ),
        )
    )
    selection = cold_verify_object_bongard_shared_witness_slate(
        select_object_bongard_shared_witness_slate(specs, spaces),
        specs,
        spaces,
    )
    support_digest = _support_digest(specs, targets, foils)
    common = {
        "runner_source_digest": object_bongard_shared_witness_task_runner_source_digest(),
        "task_plan": plan,
        "execution_precommit_digest": precommit,
        "semantic_artifact": semantic,
        "rubric_specs": specs,
        "side_0_support_by_rank": targets,
        "side_1_support_by_rank": foils,
        "support_digest": support_digest,
        "version_spaces": spaces,
        "slate_selection": selection,
        "selection_replay_calls_made": 1,
        "score_denominator": SHARED_WITNESS_TASK_SCORE_DENOMINATOR,
    }
    selected = selection.selected_candidate
    if selected is None:
        return _make_archive(
            status=_gap_status(spaces),
            **common,
            freeze=None,
            freeze_commit=None,
            side_0_query=None,
            side_1_query=None,
            query_results=(),
            correct_count=0,
            incorrect_count=2,
            abstention_count=2,
            coverage_count=0,
            accuracy_ppm=0,
            coverage_ppm=0,
            freeze_commit_calls_made=0,
            freeze_reload_calls_made=0,
            query_source_calls_made=0,
        )
    if not callable(freeze_committer):
        raise TypeError("freeze_committer must be callable")
    if not callable(freeze_reloader):
        raise TypeError("freeze_reloader must be callable")
    if not callable(query_source):
        raise TypeError("query_source must be callable")
    selected_spec = selection.selected_rubric_spec
    selected_space = selection.selected_version_space
    if selected_spec is None or selected_space is None:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "selected candidate lacks its frozen structured bindings"
        )
    freeze = ObjectBongardSharedWitnessTaskFreeze.seal(
        task_plan=plan,
        execution_precommit_digest=precommit,
        semantic_artifact=semantic,
        rubric_specs=specs,
        support_digest=support_digest,
        slate_selection=selection,
    )
    freeze_data = ObjectBongardSharedWitnessTaskFreeze.from_data(
        freeze.to_data()
    ).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, ObjectBongardSharedWitnessTaskFreezeCommit)
        else ObjectBongardSharedWitnessTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)

    # This exact durable reload is the final operation before query_source is
    # permitted to create, read, or return either query observation.
    reloaded = freeze_reloader(commit.to_data())
    if not isinstance(reloaded, bytes) or reloaded != freeze_bytes:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "durable shared-witness freeze reload differs"
        )
    try:
        decoded_reload = json.loads(reloaded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "durable freeze reload is not exact JSON"
        ) from exc
    if ObjectBongardSharedWitnessTaskFreeze.from_data(decoded_reload) != freeze:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "durable freeze object differs"
        )

    raw_queries = query_source(freeze_data, commit.to_data())
    if not isinstance(raw_queries, Mapping) or set(raw_queries) != {"side_0", "side_1"}:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "query source must return exactly side_0 and side_1"
        )
    queries = tuple(_canonical_artifact(raw_queries[side]) for side in ("side_0", "side_1"))
    if (
        tuple(item.panel_id for item in queries)
        != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
        or any(item.rubric_spec_digest != selected_spec.spec_digest for item in queries)
        or any(item.observation_context_digest != precommit for item in queries)
        or any(item.protocol_digest != selected_space.observer_protocol_digest for item in queries)
        or any(item.runtime_identity_digest != selected_space.observer_runtime_identity_digest for item in queries)
    ):
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "query artifacts differ from frozen selected spec or runtime"
        )
    evaluations = tuple(
        evaluate_object_bongard_shared_witness_candidate(selected, artifact)
        for artifact in queries
    )
    results = tuple(
        ObjectBongardSharedWitnessTaskQueryResult.create(side, artifact, evaluation)
        for side, artifact, evaluation in zip(
            ("side_0", "side_1"), queries, evaluations, strict=True
        )
    )
    correct = sum(item.correct for item in results)
    incorrect = sum(item.incorrect for item in results)
    abstained = sum(item.abstained for item in results)
    covered = sum(item.covered for item in results)
    return _make_archive(
        status=ObjectBongardSharedWitnessTaskRunStatus.COMPLETE,
        **common,
        freeze=freeze,
        freeze_commit=commit,
        side_0_query=queries[0],
        side_1_query=queries[1],
        query_results=results,
        correct_count=correct,
        incorrect_count=incorrect,
        abstention_count=abstained,
        coverage_count=covered,
        accuracy_ppm=correct * 500_000,
        coverage_ppm=covered * 500_000,
        freeze_commit_calls_made=1,
        freeze_reload_calls_made=1,
        query_source_calls_made=1,
    )


def cold_replay_object_bongard_shared_witness_task(
    archive: ObjectBongardSharedWitnessTaskRunArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> ObjectBongardSharedWitnessTaskRunArchive:
    """Replay support, selection, freeze, queries, and score without a model."""

    expected = _raw_digest(expected_archive_digest, "expected archive digest")
    supplied = (
        archive.record_digest
        if isinstance(archive, ObjectBongardSharedWitnessTaskRunArchive)
        else archive.get("record_digest")
    )
    if supplied != expected:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "shared-witness archive differs from external commitment"
        )
    restored = ObjectBongardSharedWitnessTaskRunArchive.from_data(
        archive.to_data()
        if isinstance(archive, ObjectBongardSharedWitnessTaskRunArchive)
        else archive
    )
    if restored.record_digest != expected:
        raise ObjectBongardSharedWitnessTaskRunnerError(
            "cold shared-witness archive digest differs"
        )
    return restored


__all__ = (
    "ARCHIVE_SCHEMA",
    "FREEZE_COMMIT_SCHEMA",
    "FREEZE_SCHEMA",
    "QUERY_RESULT_SCHEMA",
    "RUNNER_ID",
    "SHARED_WITNESS_TASK_SCORE_DENOMINATOR",
    "FreezeCommitter",
    "FreezeReloader",
    "ObjectBongardSharedWitnessTaskFreeze",
    "ObjectBongardSharedWitnessTaskFreezeCommit",
    "ObjectBongardSharedWitnessTaskQueryResult",
    "ObjectBongardSharedWitnessTaskRunArchive",
    "ObjectBongardSharedWitnessTaskRunStatus",
    "ObjectBongardSharedWitnessTaskRunnerError",
    "QuerySource",
    "cold_replay_object_bongard_shared_witness_task",
    "object_bongard_shared_witness_task_runner_source_digest",
    "run_object_bongard_shared_witness_task",
)
