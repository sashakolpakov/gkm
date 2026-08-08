"""Pure-Python freeze/query runner for one prose-rubric Bongard task.

The visual model supplies candidate-independent ordinal observations.  Python
derives the group-0 rubric, enumerates and verifies the complete closed
eight-predicate version space, and lets Codex rank only its survivors.  The
selected immutable candidate is then durably frozen and reloaded before the
sealed query source can be called.  Lean is absent from identity, decision,
scoring, and replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
)
from bongard.object_bongard_rubric_ranker import (
    ObjectBongardRubricRankResponse,
    object_bongard_rubric_rank_input_digest,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricCandidate,
    ObjectBongardRubricCandidateEvaluation,
    ObjectBongardRubricSupportVersionSpace,
    RubricPredicateOperator,
    RubricSupportGapKind,
    build_object_bongard_rubric_support_version_space,
    cold_verify_object_bongard_rubric_support_version_space,
    evaluate_object_bongard_rubric_candidate,
    object_bongard_rubric_version_space_algorithm_digest,
)
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUNNER_ID = "bongard.object-rubric-task/pure-python-freeze-query-v1"
FREEZE_SCHEMA = "gkm.bongard-object-rubric-task-freeze.v1"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-object-rubric-task-freeze-commit.v1"
QUERY_RESULT_SCHEMA = "gkm.bongard-object-rubric-task-query-result.v1"
ARCHIVE_SCHEMA = "gkm.bongard-object-rubric-task-run-archive.v1"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardRubricTaskRunnerError(RuntimeError):
    """A support, ranking, freeze, query, or replay boundary failed closed."""


class ObjectBongardRubricTaskRunStatus(str, Enum):
    COMPLETE = "complete"
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "disjunction_allowed": False,
        "arbitrary_predicate_code_allowed": False,
    }


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricTaskRunnerError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricTaskRunnerError(
            f"{label} must be a sha256: address"
        )
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardRubricTaskRunnerError(
            f"{label} must be a bounded identifier"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricTaskRunnerError(
            f"{label} fields differ from schema"
        )
    return value


def object_bongard_rubric_task_runner_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _canonical_parents(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    execution_precommit_digest: str,
) -> tuple[
    ObjectBongardTaskPlan,
    ObjectBongardSemanticArtifact,
    ObjectBongardRubricSpec,
    str,
]:
    if not isinstance(task_plan, ObjectBongardTaskPlan):
        raise TypeError("task_plan must be ObjectBongardTaskPlan")
    if not isinstance(semantic_artifact, ObjectBongardSemanticArtifact):
        raise TypeError("semantic_artifact must be ObjectBongardSemanticArtifact")
    plan = ObjectBongardTaskPlan.from_data(task_plan.to_data())
    semantic = ObjectBongardSemanticArtifact.from_data(
        semantic_artifact.to_data(),
        expected_artifact_digest=semantic_artifact.artifact_digest,
    )
    precommit = _address(
        execution_precommit_digest, "execution precommit digest"
    )
    if (
        semantic.status is not PrototypeSceneObserverStatus.SUCCESS
        or semantic.task_id != plan.task_id
        or semantic.group_panel_ids
        != (plan.side_0_support_panel_ids, plan.side_1_support_panel_ids)
        or semantic.observation_context_digest != precommit
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "successful semantic artifact does not bind the exact task support precommit"
        )
    spec = ObjectBongardRubricSpec.from_semantic_artifact(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )
    if spec.rubric != semantic.rubrics[0] or (
        spec.feature_nominations != semantic.feature_families[0]
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "group-0 rubric spec differs from the semantic artifact"
        )
    return plan, semantic, spec, precommit


def _canonical_artifact(
    value: ObjectBongardRubricObserverArtifact,
) -> ObjectBongardRubricObserverArtifact:
    if not isinstance(value, ObjectBongardRubricObserverArtifact):
        raise TypeError(
            "rubric evidence must contain ObjectBongardRubricObserverArtifact"
        )
    restored = ObjectBongardRubricObserverArtifact.from_data(value.to_data())
    if restored != value:
        raise ObjectBongardRubricTaskRunnerError(
            "rubric observer artifact cold round trip differs"
        )
    return restored


def _canonical_support(
    plan: ObjectBongardTaskPlan,
    spec: ObjectBongardRubricSpec,
    side_0: Sequence[ObjectBongardRubricObserverArtifact],
    side_1: Sequence[ObjectBongardRubricObserverArtifact],
) -> tuple[
    tuple[ObjectBongardRubricObserverArtifact, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
]:
    try:
        positives = tuple(
            sorted((_canonical_artifact(item) for item in side_0), key=lambda item: item.panel_id)
        )
        negatives = tuple(
            sorted((_canonical_artifact(item) for item in side_1), key=lambda item: item.panel_id)
        )
    except TypeError:
        raise
    except Exception as exc:
        raise ObjectBongardRubricTaskRunnerError(
            "support rubric artifacts are not canonical"
        ) from exc
    if (
        tuple(item.panel_id for item in positives)
        != plan.side_0_support_panel_ids
        or tuple(item.panel_id for item in negatives)
        != plan.side_1_support_panel_ids
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "support artifacts must match the exact sealed 6+6 panel identities"
        )
    all_artifacts = positives + negatives
    if (
        len(positives) != 6
        or len(negatives) != 6
        or len({item.panel_id for item in all_artifacts}) != 12
        or any(item.rubric_spec != spec for item in all_artifacts)
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "support rubric inventory or group-0 spec differs"
        )
    if len({item.catalog_digest for item in all_artifacts}) != 1:
        raise ObjectBongardRubricTaskRunnerError(
            "support artifacts must share one observer catalog identity"
        )
    if len({item.runtime_identity_digest for item in all_artifacts}) != 1:
        raise ObjectBongardRubricTaskRunnerError(
            "support artifacts must share one observer runtime identity"
        )
    return positives, negatives


def _support_digest(
    spec: ObjectBongardRubricSpec,
    positives: Sequence[ObjectBongardRubricObserverArtifact],
    negatives: Sequence[ObjectBongardRubricObserverArtifact],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-task-support.v1",
            "rubric_spec": spec.to_data(),
            "side_0_positive_artifacts": [item.to_data() for item in positives],
            "side_1_negative_artifacts": [item.to_data() for item in negatives],
            "support_labels_supplied_to_python_only": True,
            "query_material_included": False,
            **_authority_data(),
        }
    )


def _freeze_content(value: "ObjectBongardRubricTaskFreeze") -> dict[str, object]:
    return {
        "schema": FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "support_digest": value.support_digest,
        "observer_catalog_digest": value.observer_catalog_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "version_space_algorithm_digest": value.version_space_algorithm_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_input_digest": value.rank_input_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_candidate": value.selected_candidate.to_data(),
        "selected_candidate_digest": value.selected_candidate.candidate_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_formula": value.selected_formula,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_pixels_included": False,
        "query_observer_artifacts_included": False,
        "formula_frozen_before_query_source": True,
        "candidate_is_positive_at_least_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskFreeze:
    """Exact immutable decision record accepted by the release gate protocol."""

    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    semantic_artifact_digest: str
    rubric_spec_digest: str
    support_digest: str
    observer_catalog_digest: str
    observer_runtime_identity_digest: str
    version_space_algorithm_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_input_digest: str
    rank_response_digest: str
    selected_candidate: ObjectBongardRubricCandidate
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
            "semantic_artifact_digest",
            "rubric_spec_digest",
            "support_digest",
            "observer_catalog_digest",
            "observer_runtime_identity_digest",
            "version_space_algorithm_digest",
            "version_space_digest",
            "support_version_space_digest",
            "rank_input_digest",
            "rank_response_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.record_digest, "task freeze record digest")
        if not isinstance(self.selected_candidate, ObjectBongardRubricCandidate):
            raise TypeError("selected candidate has the wrong type")
        if (
            self.runner_source_digest
            != object_bongard_rubric_task_runner_source_digest()
            or self.rubric_spec_digest
            != self.selected_candidate.rubric_spec_digest
            or self.version_space_algorithm_digest
            != self.selected_candidate.algorithm_digest
            or self.version_space_algorithm_digest
            != object_bongard_rubric_version_space_algorithm_digest()
            or self.support_version_space_digest != self.version_space_digest
            or self.selected_predicate_digest
            != self.selected_candidate.candidate_digest
            or self.selected_formula != self.selected_candidate.formula
            or self.selected_candidate.operator is not RubricPredicateOperator.AT_LEAST
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(
                not isinstance(item, str) or not item
                for item in self.sealed_query_panel_ids
            )
            or self.record_digest != _content_address(_freeze_content(self))
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task freeze content differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        semantic_artifact: ObjectBongardSemanticArtifact,
        rubric_spec: ObjectBongardRubricSpec,
        support_digest: str,
        observer_catalog_digest: str,
        observer_runtime_identity_digest: str,
        version_space: ObjectBongardRubricSupportVersionSpace,
        rank_input_digest: str,
        rank_response: ObjectBongardRubricRankResponse,
        selected_candidate: ObjectBongardRubricCandidate,
    ) -> "ObjectBongardRubricTaskFreeze":
        values: dict[str, object] = {
            "runner_source_digest": object_bongard_rubric_task_runner_source_digest(),
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.record_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "semantic_artifact_digest": semantic_artifact.artifact_digest,
            "rubric_spec_digest": rubric_spec.spec_digest,
            "support_digest": support_digest,
            "observer_catalog_digest": observer_catalog_digest,
            "observer_runtime_identity_digest": observer_runtime_identity_digest,
            "version_space_algorithm_digest": version_space.algorithm_digest,
            "version_space_digest": version_space.version_space_digest,
            "support_version_space_digest": version_space.version_space_digest,
            "rank_input_digest": rank_input_digest,
            "rank_response_digest": rank_response.response_digest,
            "selected_candidate": selected_candidate,
            "selected_predicate_digest": selected_candidate.candidate_digest,
            "selected_formula": selected_candidate.formula,
            "sealed_query_panel_ids": (
                task_plan.side_0_query_panel_id,
                task_plan.side_1_query_panel_id,
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_freeze_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricTaskFreeze":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "runner_source_digest",
                "task_id",
                "task_plan_digest",
                "execution_precommit_digest",
                "semantic_artifact_digest",
                "rubric_spec_digest",
                "support_digest",
                "observer_catalog_digest",
                "observer_runtime_identity_digest",
                "version_space_algorithm_digest",
                "version_space_digest",
                "support_version_space_digest",
                "rank_input_digest",
                "rank_response_digest",
                "selected_candidate",
                "selected_candidate_digest",
                "selected_predicate_digest",
                "selected_formula",
                "sealed_query_panel_ids",
                "query_pixels_included",
                "query_observer_artifacts_included",
                "formula_frozen_before_query_source",
                "candidate_is_positive_at_least_only",
                *_authority_data(),
                "record_digest",
            },
            "rubric task freeze",
        )
        if (
            raw["schema"] != FREEZE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["query_pixels_included"] is not False
            or raw["query_observer_artifacts_included"] is not False
            or raw["formula_frozen_before_query_source"] is not True
            or raw["candidate_is_positive_at_least_only"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["sealed_query_panel_ids"], list)
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task freeze policy differs"
            )
        candidate = ObjectBongardRubricCandidate.from_data(
            raw["selected_candidate"]
        )
        if raw["selected_candidate_digest"] != candidate.candidate_digest:
            raise ObjectBongardRubricTaskRunnerError(
                "selected candidate digest differs"
            )
        result = cls(
            raw["runner_source_digest"],
            raw["task_id"],
            raw["task_plan_digest"],
            raw["execution_precommit_digest"],
            raw["semantic_artifact_digest"],
            raw["rubric_spec_digest"],
            raw["support_digest"],
            raw["observer_catalog_digest"],
            raw["observer_runtime_identity_digest"],
            raw["version_space_algorithm_digest"],
            raw["version_space_digest"],
            raw["support_version_space_digest"],
            raw["rank_input_digest"],
            raw["rank_response_digest"],
            candidate,
            raw["selected_predicate_digest"],
            raw["selected_formula"],
            tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task freeze is not canonical"
            )
        return result


def _commit_content(
    value: "ObjectBongardRubricTaskFreezeCommit",
) -> dict[str, object]:
    return {
        "schema": FREEZE_COMMIT_SCHEMA,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "durably_persisted_before_query_source": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskFreezeCommit:
    """Durable decision commitment accepted by the release gate protocol."""

    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
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
        _address(
            self.execution_precommit_digest,
            "commit execution precommit digest",
        )
        for name in (
            "version_space_digest",
            "support_version_space_digest",
            "rank_response_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        for name in (
            "task_freeze_digest",
            "exact_freeze_payload_digest",
            "task_freeze_store_receipt_digest",
            "record_digest",
        ):
            _address(getattr(self, name), name)
        if (
            self.version_space_digest != self.support_version_space_digest
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric freeze commit content differs"
            )

    @classmethod
    def seal(
        cls,
        freeze: ObjectBongardRubricTaskFreeze,
        exact_freeze_payload: bytes,
        *,
        task_freeze_store_receipt_digest: str,
    ) -> "ObjectBongardRubricTaskFreezeCommit":
        expected = canonical_json(freeze.to_data()) + b"\n"
        if exact_freeze_payload != expected:
            raise ObjectBongardRubricTaskRunnerError(
                "freeze payload bytes are not exact canonical JSON"
            )
        values: dict[str, object] = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "rank_response_digest": freeze.rank_response_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": "sha256:"
            + hashlib.sha256(expected).hexdigest(),
            "task_freeze_store_receipt_digest": _address(
                task_freeze_store_receipt_digest,
                "task freeze store receipt digest",
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_commit_content(provisional)),
        )

    def assert_matches(
        self,
        freeze: ObjectBongardRubricTaskFreeze,
        exact_freeze_payload: bytes,
    ) -> None:
        if self != type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=(
                self.task_freeze_store_receipt_digest
            ),
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric freeze commit replay differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema",
                "task_id",
                "task_plan_digest",
                "execution_precommit_digest",
                "version_space_digest",
                "support_version_space_digest",
                "rank_response_digest",
                "selected_predicate_digest",
                "task_freeze_digest",
                "exact_freeze_payload_digest",
                "task_freeze_store_receipt_digest",
                "durably_persisted_before_query_source",
                *_authority_data(),
                "record_digest",
            },
            "rubric task freeze commit",
        )
        if (
            raw["schema"] != FREEZE_COMMIT_SCHEMA
            or raw["durably_persisted_before_query_source"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric freeze commit policy differs"
            )
        result = cls(
            raw["task_id"],
            raw["task_plan_digest"],
            raw["execution_precommit_digest"],
            raw["version_space_digest"],
            raw["support_version_space_digest"],
            raw["rank_response_digest"],
            raw["selected_predicate_digest"],
            raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"],
            raw["task_freeze_store_receipt_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric freeze commit is not canonical"
            )
        return result


def _query_result_content(
    value: "ObjectBongardRubricTaskQueryResult",
) -> dict[str, object]:
    return {
        "schema": QUERY_RESULT_SCHEMA,
        "side": value.side,
        "panel_id": value.panel_id,
        "expected_disposition": value.expected_disposition.value,
        "evaluation": value.evaluation.to_data(),
        "correct": value.correct,
        "abstained": value.abstained,
        "fixed_denominator_contribution": 1,
        "abstention_counts_as_incorrect": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskQueryResult:
    side: str
    panel_id: str
    expected_disposition: Disposition
    evaluation: ObjectBongardRubricCandidateEvaluation
    correct: bool
    abstained: bool
    result_digest: str

    def __post_init__(self) -> None:
        if self.side not in ("side_0", "side_1"):
            raise ObjectBongardRubricTaskRunnerError("query side is unknown")
        expected = (
            Disposition.PRESENT
            if self.side == "side_0"
            else Disposition.CERTIFIED_ABSENT
        )
        if not isinstance(
            self.evaluation, ObjectBongardRubricCandidateEvaluation
        ):
            raise TypeError("query evaluation has the wrong type")
        if (
            self.expected_disposition is not expected
            or self.panel_id != self.evaluation.panel_id
            or self.correct is not (self.evaluation.disposition is expected)
            or self.abstained
            is not (
                self.evaluation.disposition
                in (Disposition.INDETERMINATE, Disposition.ERROR)
            )
            or self.result_digest
            != canonical_digest(_query_result_content(self))
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "query result differs from fixed scoring"
            )

    @classmethod
    def create(
        cls,
        side: str,
        artifact: ObjectBongardRubricObserverArtifact,
        evaluation: ObjectBongardRubricCandidateEvaluation,
    ) -> "ObjectBongardRubricTaskQueryResult":
        expected = (
            Disposition.PRESENT
            if side == "side_0"
            else Disposition.CERTIFIED_ABSENT
        )
        values: dict[str, object] = {
            "side": side,
            "panel_id": artifact.panel_id,
            "expected_disposition": expected,
            "evaluation": evaluation,
            "correct": evaluation.disposition is expected,
            "abstained": evaluation.disposition
            in (Disposition.INDETERMINATE, Disposition.ERROR),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            result_digest=canonical_digest(_query_result_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricTaskQueryResult":
        raw = _fields(
            value,
            {
                "schema",
                "side",
                "panel_id",
                "expected_disposition",
                "evaluation",
                "correct",
                "abstained",
                "fixed_denominator_contribution",
                "abstention_counts_as_incorrect",
                *_authority_data(),
                "result_digest",
            },
            "rubric task query result",
        )
        if (
            raw["schema"] != QUERY_RESULT_SCHEMA
            or raw["fixed_denominator_contribution"] != 1
            or raw["abstention_counts_as_incorrect"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric query result policy differs"
            )
        try:
            expected = Disposition(raw["expected_disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricTaskRunnerError(
                "query expected disposition differs"
            ) from exc
        result = cls(
            raw["side"],
            raw["panel_id"],
            expected,
            ObjectBongardRubricCandidateEvaluation.from_data(raw["evaluation"]),
            raw["correct"],
            raw["abstained"],
            raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric query result is not canonical"
            )
        return result


def _archive_content(
    value: "ObjectBongardRubricTaskRunArchive",
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
        "rubric_spec": value.rubric_spec.to_data(),
        "rubric_spec_digest": value.rubric_spec.spec_digest,
        "side_0_support": [item.to_data() for item in value.side_0_support],
        "side_1_support": [item.to_data() for item in value.side_1_support],
        "support_digest": value.support_digest,
        "version_space": value.version_space.to_data(),
        "version_space_digest": value.version_space.version_space_digest,
        "rank_input_digest": value.rank_input_digest,
        "rank_response": (
            None if value.rank_response is None else value.rank_response.to_data()
        ),
        "freeze": None if value.freeze is None else value.freeze.to_data(),
        "freeze_commit": (
            None if value.freeze_commit is None else value.freeze_commit.to_data()
        ),
        "side_0_query": (
            None if value.side_0_query is None else value.side_0_query.to_data()
        ),
        "side_1_query": (
            None if value.side_1_query is None else value.side_1_query.to_data()
        ),
        "query_results": [item.to_data() for item in value.query_results],
        "correct_count": value.correct_count,
        "abstention_count": value.abstention_count,
        "score_denominator": value.score_denominator,
        "accuracy_ppm": value.accuracy_ppm,
        "rank_calls_made": value.rank_calls_made,
        "rank_verification_calls_made": value.rank_verification_calls_made,
        "freeze_commit_calls_made": value.freeze_commit_calls_made,
        "freeze_reload_calls_made": value.freeze_reload_calls_made,
        "query_source_calls_made": value.query_source_calls_made,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskRunArchive:
    status: ObjectBongardRubricTaskRunStatus
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    semantic_artifact: ObjectBongardSemanticArtifact
    rubric_spec: ObjectBongardRubricSpec
    side_0_support: tuple[ObjectBongardRubricObserverArtifact, ...]
    side_1_support: tuple[ObjectBongardRubricObserverArtifact, ...]
    support_digest: str
    version_space: ObjectBongardRubricSupportVersionSpace
    rank_input_digest: str | None
    rank_response: ObjectBongardRubricRankResponse | None
    freeze: ObjectBongardRubricTaskFreeze | None
    freeze_commit: ObjectBongardRubricTaskFreezeCommit | None
    side_0_query: ObjectBongardRubricObserverArtifact | None
    side_1_query: ObjectBongardRubricObserverArtifact | None
    query_results: tuple[ObjectBongardRubricTaskQueryResult, ...]
    correct_count: int
    abstention_count: int
    score_denominator: int
    accuracy_ppm: int | None
    rank_calls_made: int
    rank_verification_calls_made: int
    freeze_commit_calls_made: int
    freeze_reload_calls_made: int
    query_source_calls_made: int
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObjectBongardRubricTaskRunStatus):
            raise TypeError("rubric task run status has the wrong type")
        if (
            self.runner_source_digest
            != object_bongard_rubric_task_runner_source_digest()
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "archive runner source differs"
            )
        plan, semantic, spec, precommit = _canonical_parents(
            self.task_plan,
            self.semantic_artifact,
            self.execution_precommit_digest,
        )
        positives, negatives = _canonical_support(
            plan, spec, self.side_0_support, self.side_1_support
        )
        if (
            plan != self.task_plan
            or semantic != self.semantic_artifact
            or spec != self.rubric_spec
            or precommit != self.execution_precommit_digest
            or self.support_digest != _support_digest(spec, positives, negatives)
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "archive parent or support binding differs"
            )
        version = cold_verify_object_bongard_rubric_support_version_space(
            self.version_space, spec, positives, negatives
        )
        if version != self.version_space:
            raise ObjectBongardRubricTaskRunnerError(
                "archive version-space cold replay differs"
            )
        if not version.survivor_candidate_digests:
            if version.gap is None:
                raise ObjectBongardRubricTaskRunnerError(
                    "empty version space lacks a typed support gap"
                )
            expected_status = (
                ObjectBongardRubricTaskRunStatus.LANGUAGE_GAP
                if version.gap.kind is RubricSupportGapKind.LANGUAGE_GAP
                else ObjectBongardRubricTaskRunStatus.WITNESS_GAP
            )
            if (
                self.status is not expected_status
                or any(
                    item is not None
                    for item in (
                        self.rank_input_digest,
                        self.rank_response,
                        self.freeze,
                        self.freeze_commit,
                        self.side_0_query,
                        self.side_1_query,
                        self.accuracy_ppm,
                    )
                )
                or self.query_results
                or any(
                    (
                        self.correct_count,
                        self.abstention_count,
                        self.score_denominator,
                        self.rank_calls_made,
                        self.rank_verification_calls_made,
                        self.freeze_commit_calls_made,
                        self.freeze_reload_calls_made,
                        self.query_source_calls_made,
                    )
                )
            ):
                raise ObjectBongardRubricTaskRunnerError(
                    "typed gap archive crossed a later execution phase"
                )
        else:
            self._validate_complete(
                plan=plan,
                semantic=semantic,
                spec=spec,
                precommit=precommit,
                positives=positives,
                negatives=negatives,
                version=version,
            )
        _raw_digest(self.record_digest, "archive digest")
        if self.record_digest != canonical_digest(_archive_content(self)):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task archive digest differs"
            )

    def _validate_complete(
        self,
        *,
        plan: ObjectBongardTaskPlan,
        semantic: ObjectBongardSemanticArtifact,
        spec: ObjectBongardRubricSpec,
        precommit: str,
        positives: tuple[ObjectBongardRubricObserverArtifact, ...],
        negatives: tuple[ObjectBongardRubricObserverArtifact, ...],
        version: ObjectBongardRubricSupportVersionSpace,
    ) -> None:
        if (
            self.status is not ObjectBongardRubricTaskRunStatus.COMPLETE
            or self.rank_input_digest is None
            or self.rank_response is None
            or self.freeze is None
            or self.freeze_commit is None
            or self.side_0_query is None
            or self.side_1_query is None
            or len(self.query_results) != 2
            or self.score_denominator != 2
            or self.rank_calls_made != 1
            or self.rank_verification_calls_made != 1
            or self.freeze_commit_calls_made != 1
            or self.freeze_reload_calls_made != 1
            or self.query_source_calls_made != 1
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "complete rubric archive phase counts differ"
            )
        expected_rank_input = object_bongard_rubric_rank_input_digest(
            version_space=version,
            rubric_spec=spec,
            semantic_artifact=semantic,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
        )
        self.rank_response.assert_matches(
            survivor_candidate_digests=version.survivor_candidate_digests,
            rubric_spec_digest=spec.spec_digest,
            semantic_artifact_digest=semantic.artifact_digest,
            version_space_digest=version.version_space_digest,
            rank_input_digest=expected_rank_input,
        )
        selected = version.survivor(
            self.rank_response.selected_candidate_digest
        )
        expected_freeze = ObjectBongardRubricTaskFreeze.seal(
            task_plan=plan,
            execution_precommit_digest=precommit,
            semantic_artifact=semantic,
            rubric_spec=spec,
            support_digest=self.support_digest,
            observer_catalog_digest=positives[0].catalog_digest,
            observer_runtime_identity_digest=positives[0].runtime_identity_digest,
            version_space=version,
            rank_input_digest=expected_rank_input,
            rank_response=self.rank_response,
            selected_candidate=selected,
        )
        freeze_bytes = canonical_json(expected_freeze.to_data()) + b"\n"
        self.freeze_commit.assert_matches(expected_freeze, freeze_bytes)
        queries = (self.side_0_query, self.side_1_query)
        if (
            self.rank_input_digest != expected_rank_input
            or self.freeze != expected_freeze
            or tuple(item.panel_id for item in queries)
            != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
            or any(item.rubric_spec != spec for item in queries)
            or any(
                item.catalog_digest != positives[0].catalog_digest
                or item.runtime_identity_digest
                != positives[0].runtime_identity_digest
                for item in queries
            )
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "archive rank, freeze, or sealed query binding differs"
            )
        evaluations = tuple(
            evaluate_object_bongard_rubric_candidate(selected, artifact)
            for artifact in queries
        )
        expected_results = tuple(
            ObjectBongardRubricTaskQueryResult.create(side, artifact, evaluation)
            for side, artifact, evaluation in zip(
                ("side_0", "side_1"), queries, evaluations, strict=True
            )
        )
        correct = sum(item.correct for item in expected_results)
        abstained = sum(item.abstained for item in expected_results)
        if (
            self.query_results != expected_results
            or self.correct_count != correct
            or self.abstention_count != abstained
            or self.accuracy_ppm != correct * 500_000
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "archive fixed-denominator query score differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "status",
                "runner_source_digest",
                "task_plan",
                "task_plan_digest",
                "execution_precommit_digest",
                "semantic_artifact",
                "semantic_artifact_digest",
                "rubric_spec",
                "rubric_spec_digest",
                "side_0_support",
                "side_1_support",
                "support_digest",
                "version_space",
                "version_space_digest",
                "rank_input_digest",
                "rank_response",
                "freeze",
                "freeze_commit",
                "side_0_query",
                "side_1_query",
                "query_results",
                "correct_count",
                "abstention_count",
                "score_denominator",
                "accuracy_ppm",
                "rank_calls_made",
                "rank_verification_calls_made",
                "freeze_commit_calls_made",
                "freeze_reload_calls_made",
                "query_source_calls_made",
                "cold_replay_model_calls",
                *_authority_data(),
                "record_digest",
            },
            "rubric task run archive",
        )
        if (
            raw["schema"] != ARCHIVE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[name], list)
                for name in ("side_0_support", "side_1_support", "query_results")
            )
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task archive policy differs"
            )
        plan = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        semantic = ObjectBongardSemanticArtifact.from_data(
            raw["semantic_artifact"]
        )
        spec = ObjectBongardRubricSpec.from_data(raw["rubric_spec"])
        version = ObjectBongardRubricSupportVersionSpace.from_data(
            raw["version_space"]
        )
        if (
            raw["task_plan_digest"] != plan.record_digest
            or raw["semantic_artifact_digest"] != semantic.artifact_digest
            or raw["rubric_spec_digest"] != spec.spec_digest
            or raw["version_space_digest"] != version.version_space_digest
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric archive parent digest differs"
            )
        result = cls(
            ObjectBongardRubricTaskRunStatus(raw["status"]),
            raw["runner_source_digest"],
            plan,
            raw["execution_precommit_digest"],
            semantic,
            spec,
            tuple(
                ObjectBongardRubricObserverArtifact.from_data(item)
                for item in raw["side_0_support"]
            ),
            tuple(
                ObjectBongardRubricObserverArtifact.from_data(item)
                for item in raw["side_1_support"]
            ),
            raw["support_digest"],
            version,
            raw["rank_input_digest"],
            (
                None
                if raw["rank_response"] is None
                else ObjectBongardRubricRankResponse.from_data(
                    raw["rank_response"]
                )
            ),
            (
                None
                if raw["freeze"] is None
                else ObjectBongardRubricTaskFreeze.from_data(raw["freeze"])
            ),
            (
                None
                if raw["freeze_commit"] is None
                else ObjectBongardRubricTaskFreezeCommit.from_data(
                    raw["freeze_commit"]
                )
            ),
            (
                None
                if raw["side_0_query"] is None
                else ObjectBongardRubricObserverArtifact.from_data(
                    raw["side_0_query"]
                )
            ),
            (
                None
                if raw["side_1_query"] is None
                else ObjectBongardRubricObserverArtifact.from_data(
                    raw["side_1_query"]
                )
            ),
            tuple(
                ObjectBongardRubricTaskQueryResult.from_data(item)
                for item in raw["query_results"]
            ),
            raw["correct_count"],
            raw["abstention_count"],
            raw["score_denominator"],
            raw["accuracy_ppm"],
            raw["rank_calls_made"],
            raw["rank_verification_calls_made"],
            raw["freeze_commit_calls_made"],
            raw["freeze_reload_calls_made"],
            raw["query_source_calls_made"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError(
                "rubric task archive is not canonical"
            )
        return result


class ObjectBongardRubricTaskRanker(Protocol):
    def __call__(
        self,
        version_space: ObjectBongardRubricSupportVersionSpace,
        **kwargs: object,
    ) -> object: ...

    def verify_response(
        self, response: ObjectBongardRubricRankResponse, **kwargs: object
    ) -> object: ...


FreezeCommitter = Callable[
    [bytes], ObjectBongardRubricTaskFreezeCommit | Mapping[str, Any]
]
FreezeReloader = Callable[[Mapping[str, object]], bytes]
QuerySource = Callable[
    [Mapping[str, object], Mapping[str, object]],
    Mapping[str, ObjectBongardRubricObserverArtifact],
]


def _make_archive(**values: object) -> ObjectBongardRubricTaskRunArchive:
    provisional = object.__new__(ObjectBongardRubricTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricTaskRunArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_content(provisional)),
    )


def run_object_bongard_rubric_task(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    side_0_support: Sequence[ObjectBongardRubricObserverArtifact],
    side_1_support: Sequence[ObjectBongardRubricObserverArtifact],
    *,
    execution_precommit_digest: str,
    ranker: ObjectBongardRubricTaskRanker,
    freeze_committer: FreezeCommitter,
    freeze_reloader: FreezeReloader,
    query_source: QuerySource,
) -> ObjectBongardRubricTaskRunArchive:
    """Run exact 6+6 support through one durably frozen 1+1 query release."""

    plan, semantic, spec, precommit = _canonical_parents(
        task_plan, semantic_artifact, execution_precommit_digest
    )
    positives, negatives = _canonical_support(
        plan, spec, side_0_support, side_1_support
    )
    version = build_object_bongard_rubric_support_version_space(
        spec, positives, negatives
    )
    version = cold_verify_object_bongard_rubric_support_version_space(
        version, spec, positives, negatives
    )
    support_digest = _support_digest(spec, positives, negatives)
    common: dict[str, object] = {
        "runner_source_digest": object_bongard_rubric_task_runner_source_digest(),
        "task_plan": plan,
        "execution_precommit_digest": precommit,
        "semantic_artifact": semantic,
        "rubric_spec": spec,
        "side_0_support": positives,
        "side_1_support": negatives,
        "support_digest": support_digest,
        "version_space": version,
    }
    if not version.survivor_candidate_digests:
        if version.gap is None:
            raise ObjectBongardRubricTaskRunnerError(
                "empty rubric version space lacks a typed gap"
            )
        return _make_archive(
            status=(
                ObjectBongardRubricTaskRunStatus.LANGUAGE_GAP
                if version.gap.kind is RubricSupportGapKind.LANGUAGE_GAP
                else ObjectBongardRubricTaskRunStatus.WITNESS_GAP
            ),
            **common,
            rank_input_digest=None,
            rank_response=None,
            freeze=None,
            freeze_commit=None,
            side_0_query=None,
            side_1_query=None,
            query_results=(),
            correct_count=0,
            abstention_count=0,
            score_denominator=0,
            accuracy_ppm=None,
            rank_calls_made=0,
            rank_verification_calls_made=0,
            freeze_commit_calls_made=0,
            freeze_reload_calls_made=0,
            query_source_calls_made=0,
        )
    rank_input = object_bongard_rubric_rank_input_digest(
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
    )
    raw_response = ranker(
        version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )
    response = (
        raw_response
        if isinstance(raw_response, ObjectBongardRubricRankResponse)
        else ObjectBongardRubricRankResponse.from_data(raw_response)
    )
    ranker.verify_response(
        response,
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
        expected_response_digest=response.response_digest,
    )
    response.assert_matches(
        survivor_candidate_digests=version.survivor_candidate_digests,
        rubric_spec_digest=spec.spec_digest,
        semantic_artifact_digest=semantic.artifact_digest,
        version_space_digest=version.version_space_digest,
        rank_input_digest=rank_input,
    )
    selected = version.survivor(response.selected_candidate_digest)
    freeze = ObjectBongardRubricTaskFreeze.seal(
        task_plan=plan,
        execution_precommit_digest=precommit,
        semantic_artifact=semantic,
        rubric_spec=spec,
        support_digest=support_digest,
        observer_catalog_digest=positives[0].catalog_digest,
        observer_runtime_identity_digest=positives[0].runtime_identity_digest,
        version_space=version,
        rank_input_digest=rank_input,
        rank_response=response,
        selected_candidate=selected,
    )
    freeze_data = ObjectBongardRubricTaskFreeze.from_data(
        freeze.to_data()
    ).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, ObjectBongardRubricTaskFreezeCommit)
        else ObjectBongardRubricTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)

    # This reload is mandatory.  The query callback is intentionally below it.
    reloaded = freeze_reloader(commit.to_data())
    if reloaded != freeze_bytes:
        raise ObjectBongardRubricTaskRunnerError(
            "durable rubric freeze reload differs"
        )
    try:
        decoded_reload = json.loads(reloaded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricTaskRunnerError(
            "durable rubric freeze reload is not JSON"
        ) from exc
    if ObjectBongardRubricTaskFreeze.from_data(decoded_reload) != freeze:
        raise ObjectBongardRubricTaskRunnerError(
            "durable rubric freeze object differs"
        )

    raw_queries = query_source(freeze_data, commit.to_data())
    if (
        not isinstance(raw_queries, Mapping)
        or set(raw_queries) != {"side_0", "side_1"}
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "query source must return exactly side_0 and side_1"
        )
    queries = tuple(
        _canonical_artifact(raw_queries[side])
        for side in ("side_0", "side_1")
    )
    if (
        tuple(item.panel_id for item in queries)
        != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
        or any(item.rubric_spec != spec for item in queries)
        or any(
            item.catalog_digest != positives[0].catalog_digest
            or item.runtime_identity_digest != positives[0].runtime_identity_digest
            for item in queries
        )
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "query rubric artifacts differ from the sealed identities or runtime"
        )
    evaluations = tuple(
        evaluate_object_bongard_rubric_candidate(selected, artifact)
        for artifact in queries
    )
    results = tuple(
        ObjectBongardRubricTaskQueryResult.create(side, artifact, evaluation)
        for side, artifact, evaluation in zip(
            ("side_0", "side_1"), queries, evaluations, strict=True
        )
    )
    correct = sum(item.correct for item in results)
    return _make_archive(
        status=ObjectBongardRubricTaskRunStatus.COMPLETE,
        **common,
        rank_input_digest=rank_input,
        rank_response=response,
        freeze=freeze,
        freeze_commit=commit,
        side_0_query=queries[0],
        side_1_query=queries[1],
        query_results=results,
        correct_count=correct,
        abstention_count=sum(item.abstained for item in results),
        score_denominator=2,
        accuracy_ppm=correct * 500_000,
        rank_calls_made=1,
        rank_verification_calls_made=1,
        freeze_commit_calls_made=1,
        freeze_reload_calls_made=1,
        query_source_calls_made=1,
    )


def cold_replay_object_bongard_rubric_task(
    archive: ObjectBongardRubricTaskRunArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> ObjectBongardRubricTaskRunArchive:
    """Replay all Python decisions without rank, observer, query, or model calls."""

    expected = _raw_digest(expected_archive_digest, "expected archive digest")
    supplied = (
        archive.record_digest
        if isinstance(archive, ObjectBongardRubricTaskRunArchive)
        else archive.get("record_digest")
    )
    if supplied != expected:
        raise ObjectBongardRubricTaskRunnerError(
            "rubric task archive differs from external commitment"
        )
    restored = (
        ObjectBongardRubricTaskRunArchive.from_data(archive.to_data())
        if isinstance(archive, ObjectBongardRubricTaskRunArchive)
        else ObjectBongardRubricTaskRunArchive.from_data(archive)
    )
    if restored.record_digest != expected:
        raise ObjectBongardRubricTaskRunnerError(
            "cold rubric task archive digest differs"
        )
    return restored


__all__ = (
    "ARCHIVE_SCHEMA",
    "FREEZE_COMMIT_SCHEMA",
    "FREEZE_SCHEMA",
    "QUERY_RESULT_SCHEMA",
    "RUNNER_ID",
    "ObjectBongardRubricTaskFreeze",
    "ObjectBongardRubricTaskFreezeCommit",
    "ObjectBongardRubricTaskQueryResult",
    "ObjectBongardRubricTaskRunArchive",
    "ObjectBongardRubricTaskRunStatus",
    "ObjectBongardRubricTaskRunnerError",
    "cold_replay_object_bongard_rubric_task",
    "object_bongard_rubric_task_runner_source_digest",
    "run_object_bongard_rubric_task",
)
