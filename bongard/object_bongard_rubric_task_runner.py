"""Pure-Python two-rank freeze/query runner for one Bongard task.

Vision proposes exactly two ranked positive soft-cue pairs from all six
support panels on each side.  The observer then produces candidate-independent
four-disposition evidence for both derived rubric specs.  Python evaluates the
fixed rank-major slate (rank-0 OBJECT, rank-0 SCENE, rank-1 OBJECT, rank-1
SCENE), freezes its first exact support survivor, durably reloads that freeze,
and only then permits the query source to load query pixels.  No model ranks
survivors and Lean is absent from identity, decision, scoring, and replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
)
from bongard.object_bongard_rubric_slate import (
    ObjectBongardRubricSlateSelection,
    cold_verify_object_bongard_rubric_slate,
    object_bongard_rubric_slate_algorithm_digest,
    select_object_bongard_rubric_slate,
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
)
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUNNER_ID = "bongard.object-rubric-task/two-rank-python-slate-freeze-query-v2"
FREEZE_SCHEMA = "gkm.bongard-object-rubric-task-freeze.v2"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-object-rubric-task-freeze-commit.v2"
QUERY_RESULT_SCHEMA = "gkm.bongard-object-rubric-task-query-result.v2"
ARCHIVE_SCHEMA = "gkm.bongard-object-rubric-task-run-archive.v2"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardRubricTaskRunnerError(RuntimeError):
    """A support, selection, freeze, query, or replay boundary failed closed."""


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
        "threshold_tuning_allowed": False,
        "model_selects_scope_or_candidate": False,
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
    tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
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
    specs = tuple(
        ObjectBongardRubricSpec.from_semantic_artifact(
            semantic,
            expected_artifact_digest=semantic.artifact_digest,
            candidate_rank=rank,
        )
        for rank in (0, 1)
    )
    if (
        len(semantic.soft_cue_candidates) != 2
        or tuple(item.candidate_rank for item in semantic.soft_cue_candidates)
        != (0, 1)
        or tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != 2
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "semantic artifact must bind two distinct ranked soft-cue rubric specs"
        )
    return plan, semantic, specs, precommit


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
    specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
    side_0: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
    side_1: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
) -> tuple[
    tuple[
        tuple[ObjectBongardRubricObserverArtifact, ...],
        tuple[ObjectBongardRubricObserverArtifact, ...],
    ],
    tuple[
        tuple[ObjectBongardRubricObserverArtifact, ...],
        tuple[ObjectBongardRubricObserverArtifact, ...],
    ],
]:
    try:
        positives = tuple(
            tuple(
                sorted(
                    (_canonical_artifact(item) for item in block),
                    key=lambda item: item.panel_id,
                )
            )
            for block in side_0
        )
        negatives = tuple(
            tuple(
                sorted(
                    (_canonical_artifact(item) for item in block),
                    key=lambda item: item.panel_id,
                )
            )
            for block in side_1
        )
    except TypeError:
        raise
    except Exception as exc:
        raise ObjectBongardRubricTaskRunnerError(
            "support rubric artifacts are not canonical"
        ) from exc
    if len(positives) != 2 or len(negatives) != 2:
        raise ObjectBongardRubricTaskRunnerError(
            "support evidence must contain exactly two ranked spec blocks"
        )
    if any(
        tuple(item.panel_id for item in block) != expected
        for blocks, expected in (
            (positives, plan.side_0_support_panel_ids),
            (negatives, plan.side_1_support_panel_ids),
        )
        for block in blocks
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "each ranked spec must observe the exact sealed 6+6 support panels"
        )
    all_artifacts = tuple(
        item for blocks in (positives, negatives) for block in blocks for item in block
    )
    if (
        any(len(block) != 6 for block in (*positives, *negatives))
        or len(all_artifacts) != 24
        or any(
            item.rubric_spec != specs[rank]
            for rank in (0, 1)
            for block in (positives[rank], negatives[rank])
            for item in block
        )
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "two-rank support rubric inventory or spec binding differs"
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
    specs: Sequence[ObjectBongardRubricSpec],
    positives: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
    negatives: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-task-support.v2",
            "rubric_specs": [item.to_data() for item in specs],
            "side_0_positive_artifacts_by_rank": [
                [item.to_data() for item in block] for block in positives
            ],
            "side_1_negative_artifacts_by_rank": [
                [item.to_data() for item in block] for block in negatives
            ],
            "candidate_rank_order": [0, 1],
            "support_panels_per_side_per_spec": 6,
            "support_labels_supplied_to_python_only": True,
            "query_material_included": False,
            **_authority_data(),
        }
    )


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


# ---------------------------------------------------------------------------
# Two-rank v2 records.  Old single-spec archives remain readable only by their
# original committed source revision; this source exposes only the deterministic
# Python slate path.


def _freeze_v2_content(value: "ObjectBongardRubricTaskFreeze") -> dict[str, object]:
    selection = value.slate_selection
    selected = selection.selected_candidate
    if selected is None:  # pragma: no cover - constructor rejects this
        raise ObjectBongardRubricTaskRunnerError("cannot serialize an empty freeze")
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
        "observer_catalog_digest": value.observer_catalog_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "slate_algorithm_digest": value.slate_algorithm_digest,
        "version_space_digests": list(value.version_space_digests),
        "slate_selection": selection.to_data(),
        "slate_selection_digest": selection.selection_digest,
        # The release gate's protocol predates deterministic slate selection.
        # These three aliases contain the exact slate-selection commitment;
        # they do not represent a model rank response.
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_response_digest_is_legacy_alias_for_slate_selection": True,
        "selected_candidate": selected.to_data(),
        "selected_candidate_digest": selected.candidate_digest,
        "selected_rubric_spec_digest": selected.rubric_spec_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_formula": value.selected_formula,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_pixels_included": False,
        "query_observer_artifacts_included": False,
        "formula_frozen_before_query_source": True,
        "two_rank_support_slate_frozen": True,
        "candidate_order": [
            "rank-0/object", "rank-0/scene", "rank-1/object", "rank-1/scene"
        ],
        "candidate_is_positive_at_least_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskFreeze:
    """Durable two-spec Python selection committed before query release."""

    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    support_digest: str
    observer_catalog_digest: str
    observer_runtime_identity_digest: str
    slate_algorithm_digest: str
    version_space_digests: tuple[str, str]
    slate_selection: ObjectBongardRubricSlateSelection
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
            "semantic_artifact_digest", "support_digest",
            "observer_catalog_digest", "observer_runtime_identity_digest",
            "slate_algorithm_digest", "version_space_digest",
            "support_version_space_digest", "rank_response_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.record_digest, "task freeze record digest")
        if (
            not isinstance(self.rubric_specs, tuple)
            or len(self.rubric_specs) != 2
            or any(not isinstance(item, ObjectBongardRubricSpec) for item in self.rubric_specs)
        ):
            raise ObjectBongardRubricTaskRunnerError("freeze requires two rubric specs")
        specs = tuple(ObjectBongardRubricSpec.from_data(item.to_data()) for item in self.rubric_specs)
        if specs != self.rubric_specs or tuple(item.candidate_rank for item in specs) != (0, 1):
            raise ObjectBongardRubricTaskRunnerError("freeze rubric rank order differs")
        if not isinstance(self.slate_selection, ObjectBongardRubricSlateSelection):
            raise TypeError("freeze slate selection has the wrong type")
        selection = ObjectBongardRubricSlateSelection.from_data(self.slate_selection.to_data())
        selected = selection.selected_candidate
        if selected is None:
            raise ObjectBongardRubricTaskRunnerError("freeze cannot bind an empty slate")
        spaces = selection.version_spaces
        if (
            self.runner_source_digest != object_bongard_rubric_task_runner_source_digest()
            or selection != self.slate_selection
            or selection.rubric_specs != specs
            or self.semantic_artifact_digest != selection.semantic_artifact_digest
            or self.slate_algorithm_digest != object_bongard_rubric_slate_algorithm_digest()
            or self.slate_algorithm_digest != selection.algorithm_digest
            or self.version_space_digests
            != tuple(item.version_space_digest for item in spaces)
            or self.observer_catalog_digest != spaces[0].observer_catalog_digest
            or self.observer_runtime_identity_digest
            != spaces[0].observer_runtime_identity_digest
            or self.version_space_digest != selection.selection_digest
            or self.support_version_space_digest != selection.selection_digest
            or self.rank_response_digest != selection.selection_digest
            or self.selected_predicate_digest != selected.candidate_digest
            or self.selected_formula != selected.formula
            or selected.operator is not RubricPredicateOperator.AT_LEAST
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(not isinstance(item, str) or not item for item in self.sealed_query_panel_ids)
            or self.record_digest != _content_address(_freeze_v2_content(self))
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank task freeze content differs")

    @property
    def selected_candidate(self) -> ObjectBongardRubricCandidate:
        selected = self.slate_selection.selected_candidate
        if selected is None:  # pragma: no cover - guarded by __post_init__
            raise ObjectBongardRubricTaskRunnerError("freeze selection is empty")
        return selected

    @property
    def selected_rubric_spec(self) -> ObjectBongardRubricSpec:
        return next(
            item for item in self.rubric_specs
            if item.spec_digest == self.selected_candidate.rubric_spec_digest
        )

    @property
    def rubric_spec_digest(self) -> str:
        """Compatibility accessor for the selected query rubric only."""
        return self.selected_rubric_spec.spec_digest

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        semantic_artifact: ObjectBongardSemanticArtifact,
        rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
        support_digest: str,
        slate_selection: ObjectBongardRubricSlateSelection,
    ) -> "ObjectBongardRubricTaskFreeze":
        selected = slate_selection.selected_candidate
        if selected is None:
            raise ObjectBongardRubricTaskRunnerError("cannot freeze a slate without a survivor")
        spaces = slate_selection.version_spaces
        values: dict[str, object] = {
            "runner_source_digest": object_bongard_rubric_task_runner_source_digest(),
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.record_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "semantic_artifact_digest": semantic_artifact.artifact_digest,
            "rubric_specs": rubric_specs,
            "support_digest": support_digest,
            "observer_catalog_digest": spaces[0].observer_catalog_digest,
            "observer_runtime_identity_digest": spaces[0].observer_runtime_identity_digest,
            "slate_algorithm_digest": slate_selection.algorithm_digest,
            "version_space_digests": tuple(item.version_space_digest for item in spaces),
            "slate_selection": slate_selection,
            "version_space_digest": slate_selection.selection_digest,
            "support_version_space_digest": slate_selection.selection_digest,
            "rank_response_digest": slate_selection.selection_digest,
            "selected_predicate_digest": selected.candidate_digest,
            "selected_formula": selected.formula,
            "sealed_query_panel_ids": (
                task_plan.side_0_query_panel_id, task_plan.side_1_query_panel_id
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_freeze_v2_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_v2_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact_digest", "rubric_specs", "rubric_spec_digests",
                "support_digest", "observer_catalog_digest",
                "observer_runtime_identity_digest", "slate_algorithm_digest",
                "version_space_digests", "slate_selection", "slate_selection_digest",
                "version_space_digest", "support_version_space_digest",
                "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_candidate", "selected_candidate_digest",
                "selected_rubric_spec_digest", "selected_predicate_digest",
                "selected_formula", "sealed_query_panel_ids", "query_pixels_included",
                "query_observer_artifacts_included", "formula_frozen_before_query_source",
                "two_rank_support_slate_frozen", "candidate_order",
                "candidate_is_positive_at_least_only", *_authority_data(), "record_digest",
            },
            "two-rank rubric task freeze",
        )
        if any(not isinstance(raw[name], list) for name in (
            "rubric_specs", "rubric_spec_digests", "version_space_digests",
            "sealed_query_panel_ids", "candidate_order",
        )):
            raise ObjectBongardRubricTaskRunnerError("freeze arrays are malformed")
        specs = tuple(ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"])
        selection = ObjectBongardRubricSlateSelection.from_data(raw["slate_selection"])
        selected = selection.selected_candidate
        if (
            raw["schema"] != FREEZE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["slate_selection_digest"] != selection.selection_digest
            or selected is None
            or raw["selected_candidate"] != selected.to_data()
            or raw["selected_candidate_digest"] != selected.candidate_digest
            or raw["selected_rubric_spec_digest"] != selected.rubric_spec_digest
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["query_pixels_included"] is not False
            or raw["query_observer_artifacts_included"] is not False
            or raw["formula_frozen_before_query_source"] is not True
            or raw["two_rank_support_slate_frozen"] is not True
            or raw["candidate_order"] != [
                "rank-0/object", "rank-0/scene", "rank-1/object", "rank-1/scene"
            ]
            or raw["candidate_is_positive_at_least_only"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze policy differs")
        result = cls(
            runner_source_digest=raw["runner_source_digest"],
            task_id=raw["task_id"],
            task_plan_digest=raw["task_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            semantic_artifact_digest=raw["semantic_artifact_digest"],
            rubric_specs=specs,  # type: ignore[arg-type]
            support_digest=raw["support_digest"],
            observer_catalog_digest=raw["observer_catalog_digest"],
            observer_runtime_identity_digest=raw["observer_runtime_identity_digest"],
            slate_algorithm_digest=raw["slate_algorithm_digest"],
            version_space_digests=tuple(raw["version_space_digests"]),  # type: ignore[arg-type]
            slate_selection=selection,
            version_space_digest=raw["version_space_digest"],
            support_version_space_digest=raw["support_version_space_digest"],
            rank_response_digest=raw["rank_response_digest"],
            selected_predicate_digest=raw["selected_predicate_digest"],
            selected_formula=raw["selected_formula"],
            sealed_query_panel_ids=tuple(raw["sealed_query_panel_ids"]),  # type: ignore[arg-type]
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze is not canonical")
        return result


def _commit_v2_content(value: "ObjectBongardRubricTaskFreezeCommit") -> dict[str, object]:
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
        "durably_persisted_before_query_source": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskFreezeCommit:
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
        _address(self.execution_precommit_digest, "commit execution precommit digest")
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
            or self.record_digest != _content_address(_commit_v2_content(self))
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze commit differs")

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
            raise ObjectBongardRubricTaskRunnerError("freeze payload bytes are not exact canonical JSON")
        values: dict[str, object] = {
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
                task_freeze_store_receipt_digest, "task freeze store receipt digest"
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_commit_v2_content(provisional)),
        )

    def assert_matches(self, freeze: ObjectBongardRubricTaskFreeze, exact_freeze_payload: bytes) -> None:
        if self != type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=self.task_freeze_store_receipt_digest,
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze commit replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_commit_v2_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "task_id", "task_plan_digest", "execution_precommit_digest",
                "slate_selection_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "task_freeze_store_receipt_digest",
                "durably_persisted_before_query_source", *_authority_data(), "record_digest",
            },
            "two-rank rubric task freeze commit",
        )
        if (
            raw["schema"] != FREEZE_COMMIT_SCHEMA
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["durably_persisted_before_query_source"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze commit policy differs")
        result = cls(
            task_id=raw["task_id"],
            task_plan_digest=raw["task_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            slate_selection_digest=raw["slate_selection_digest"],
            version_space_digest=raw["version_space_digest"],
            support_version_space_digest=raw["support_version_space_digest"],
            rank_response_digest=raw["rank_response_digest"],
            selected_predicate_digest=raw["selected_predicate_digest"],
            task_freeze_digest=raw["task_freeze_digest"],
            exact_freeze_payload_digest=raw["exact_freeze_payload_digest"],
            task_freeze_store_receipt_digest=raw["task_freeze_store_receipt_digest"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError("two-rank freeze commit is not canonical")
        return result


def _archive_v2_content(value: "ObjectBongardRubricTaskRunArchive") -> dict[str, object]:
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
        "abstention_count": value.abstention_count,
        "score_denominator": value.score_denominator,
        "accuracy_ppm": value.accuracy_ppm,
        "freeze_commit_calls_made": value.freeze_commit_calls_made,
        "freeze_reload_calls_made": value.freeze_reload_calls_made,
        "query_source_calls_made": value.query_source_calls_made,
        "support_spec_count": 2,
        "support_observations_per_spec": 12,
        "candidate_order": [
            "rank-0/object", "rank-0/scene", "rank-1/object", "rank-1/scene"
        ],
        "query_uses_selected_python_candidate_only": True,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


def _slate_gap_status(
    spaces: Sequence[ObjectBongardRubricSupportVersionSpace],
) -> ObjectBongardRubricTaskRunStatus:
    if any(item.survivor_candidate_digests for item in spaces):
        raise ObjectBongardRubricTaskRunnerError(
            "empty slate cannot contain a support survivor"
        )
    if any(item.gap is None for item in spaces):
        raise ObjectBongardRubricTaskRunnerError(
            "empty slate version space lacks a typed gap"
        )
    return (
        ObjectBongardRubricTaskRunStatus.WITNESS_GAP
        if any(item.gap.kind is RubricSupportGapKind.WITNESS_GAP for item in spaces)  # type: ignore[union-attr]
        else ObjectBongardRubricTaskRunStatus.LANGUAGE_GAP
    )


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskRunArchive:
    status: ObjectBongardRubricTaskRunStatus
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    semantic_artifact: ObjectBongardSemanticArtifact
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    side_0_support_by_rank: tuple[
        tuple[ObjectBongardRubricObserverArtifact, ...],
        tuple[ObjectBongardRubricObserverArtifact, ...],
    ]
    side_1_support_by_rank: tuple[
        tuple[ObjectBongardRubricObserverArtifact, ...],
        tuple[ObjectBongardRubricObserverArtifact, ...],
    ]
    support_digest: str
    version_spaces: tuple[
        ObjectBongardRubricSupportVersionSpace,
        ObjectBongardRubricSupportVersionSpace,
    ]
    slate_selection: ObjectBongardRubricSlateSelection
    freeze: ObjectBongardRubricTaskFreeze | None
    freeze_commit: ObjectBongardRubricTaskFreezeCommit | None
    side_0_query: ObjectBongardRubricObserverArtifact | None
    side_1_query: ObjectBongardRubricObserverArtifact | None
    query_results: tuple[ObjectBongardRubricTaskQueryResult, ...]
    correct_count: int
    abstention_count: int
    score_denominator: int
    accuracy_ppm: int | None
    selection_replay_calls_made: int
    freeze_commit_calls_made: int
    freeze_reload_calls_made: int
    query_source_calls_made: int
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObjectBongardRubricTaskRunStatus):
            raise TypeError("rubric task run status has the wrong type")
        if self.runner_source_digest != object_bongard_rubric_task_runner_source_digest():
            raise ObjectBongardRubricTaskRunnerError("archive runner source differs")
        plan, semantic, specs, precommit = _canonical_parents(
            self.task_plan, self.semantic_artifact, self.execution_precommit_digest
        )
        positives, negatives = _canonical_support(
            plan, specs, self.side_0_support_by_rank, self.side_1_support_by_rank
        )
        support_digest = _support_digest(specs, positives, negatives)
        if (
            plan != self.task_plan
            or semantic != self.semantic_artifact
            or specs != self.rubric_specs
            or precommit != self.execution_precommit_digest
            or positives != self.side_0_support_by_rank
            or negatives != self.side_1_support_by_rank
            or support_digest != self.support_digest
            or not isinstance(self.version_spaces, tuple)
            or len(self.version_spaces) != 2
        ):
            raise ObjectBongardRubricTaskRunnerError("archive parent or support slate differs")
        replayed_spaces = tuple(
            cold_verify_object_bongard_rubric_support_version_space(
                self.version_spaces[rank], specs[rank], positives[rank], negatives[rank]
            )
            for rank in (0, 1)
        )
        if replayed_spaces != self.version_spaces:
            raise ObjectBongardRubricTaskRunnerError("archive version-space replay differs")
        selection = cold_verify_object_bongard_rubric_slate(
            self.slate_selection, specs, replayed_spaces
        )
        if selection != self.slate_selection or self.selection_replay_calls_made != 1:
            raise ObjectBongardRubricTaskRunnerError("archive slate replay differs")
        if selection.selected_candidate is None:
            expected_status = _slate_gap_status(replayed_spaces)
            if (
                self.status is not expected_status
                or any(item is not None for item in (
                    self.freeze, self.freeze_commit, self.side_0_query,
                    self.side_1_query, self.accuracy_ppm,
                ))
                or self.query_results
                or any((
                    self.correct_count, self.abstention_count, self.score_denominator,
                    self.freeze_commit_calls_made, self.freeze_reload_calls_made,
                    self.query_source_calls_made,
                ))
            ):
                raise ObjectBongardRubricTaskRunnerError(
                    "typed two-rank gap crossed a later execution phase"
                )
        else:
            self._validate_complete(
                plan=plan,
                semantic=semantic,
                specs=specs,
                precommit=precommit,
                positives=positives,
                negatives=negatives,
                selection=selection,
            )
        _raw_digest(self.record_digest, "archive digest")
        if self.record_digest != canonical_digest(_archive_v2_content(self)):
            raise ObjectBongardRubricTaskRunnerError("two-rank task archive digest differs")

    @property
    def selected_candidate(self) -> ObjectBongardRubricCandidate | None:
        return self.slate_selection.selected_candidate

    @property
    def selected_rubric_spec(self) -> ObjectBongardRubricSpec | None:
        selected = self.selected_candidate
        if selected is None:
            return None
        return next(
            item for item in self.rubric_specs
            if item.spec_digest == selected.rubric_spec_digest
        )

    def _validate_complete(
        self,
        *,
        plan: ObjectBongardTaskPlan,
        semantic: ObjectBongardSemanticArtifact,
        specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
        precommit: str,
        positives: tuple[
            tuple[ObjectBongardRubricObserverArtifact, ...],
            tuple[ObjectBongardRubricObserverArtifact, ...],
        ],
        negatives: tuple[
            tuple[ObjectBongardRubricObserverArtifact, ...],
            tuple[ObjectBongardRubricObserverArtifact, ...],
        ],
        selection: ObjectBongardRubricSlateSelection,
    ) -> None:
        selected = selection.selected_candidate
        selected_spec = self.selected_rubric_spec
        if (
            selected is None
            or selected_spec is None
            or self.status is not ObjectBongardRubricTaskRunStatus.COMPLETE
            or self.freeze is None
            or self.freeze_commit is None
            or self.side_0_query is None
            or self.side_1_query is None
            or len(self.query_results) != 2
            or self.score_denominator != 2
            or self.freeze_commit_calls_made != 1
            or self.freeze_reload_calls_made != 1
            or self.query_source_calls_made != 1
        ):
            raise ObjectBongardRubricTaskRunnerError("complete two-rank archive phase counts differ")
        expected_freeze = ObjectBongardRubricTaskFreeze.seal(
            task_plan=plan,
            execution_precommit_digest=precommit,
            semantic_artifact=semantic,
            rubric_specs=specs,
            support_digest=self.support_digest,
            slate_selection=selection,
        )
        freeze_bytes = canonical_json(expected_freeze.to_data()) + b"\n"
        self.freeze_commit.assert_matches(expected_freeze, freeze_bytes)
        queries = (self.side_0_query, self.side_1_query)
        if (
            self.freeze != expected_freeze
            or tuple(item.panel_id for item in queries)
            != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
            or any(item.rubric_spec != selected_spec for item in queries)
            or any(
                item.catalog_digest != positives[0][0].catalog_digest
                or item.runtime_identity_digest != positives[0][0].runtime_identity_digest
                for item in queries
            )
        ):
            raise ObjectBongardRubricTaskRunnerError(
                "archive freeze or selected-spec query binding differs"
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
            raise ObjectBongardRubricTaskRunnerError("archive fixed-denominator score differs")

    def to_data(self) -> dict[str, object]:
        return {**_archive_v2_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "status", "runner_source_digest",
                "task_plan", "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact", "semantic_artifact_digest", "rubric_specs",
                "rubric_spec_digests", "side_0_support_by_rank",
                "side_1_support_by_rank", "support_digest", "version_spaces",
                "version_space_digests", "slate_selection", "slate_selection_digest",
                "selection_model_calls_made", "selection_replay_calls_made",
                "freeze", "freeze_commit", "side_0_query", "side_1_query",
                "query_results", "correct_count", "abstention_count",
                "score_denominator", "accuracy_ppm", "freeze_commit_calls_made",
                "freeze_reload_calls_made", "query_source_calls_made",
                "support_spec_count", "support_observations_per_spec",
                "candidate_order", "query_uses_selected_python_candidate_only",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "two-rank rubric task archive",
        )
        for name in (
            "rubric_specs", "rubric_spec_digests", "side_0_support_by_rank",
            "side_1_support_by_rank", "version_spaces", "version_space_digests",
            "query_results", "candidate_order",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardRubricTaskRunnerError(f"{name} must be a JSON list")
        plan = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        semantic = ObjectBongardSemanticArtifact.from_data(raw["semantic_artifact"])
        specs = tuple(ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"])
        spaces = tuple(
            ObjectBongardRubricSupportVersionSpace.from_data(item)
            for item in raw["version_spaces"]
        )
        selection = ObjectBongardRubricSlateSelection.from_data(raw["slate_selection"])
        if (
            raw["schema"] != ARCHIVE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["task_plan_digest"] != plan.record_digest
            or raw["semantic_artifact_digest"] != semantic.artifact_digest
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["version_space_digests"] != [item.version_space_digest for item in spaces]
            or raw["slate_selection_digest"] != selection.selection_digest
            or raw["selection_model_calls_made"] != 0
            or raw["support_spec_count"] != 2
            or raw["support_observations_per_spec"] != 12
            or raw["candidate_order"] != [
                "rank-0/object", "rank-0/scene", "rank-1/object", "rank-1/scene"
            ]
            or raw["query_uses_selected_python_candidate_only"] is not True
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(not isinstance(block, list) for name in (
                "side_0_support_by_rank", "side_1_support_by_rank"
            ) for block in raw[name])
        ):
            raise ObjectBongardRubricTaskRunnerError("two-rank archive policy differs")
        result = cls(
            status=ObjectBongardRubricTaskRunStatus(raw["status"]),
            runner_source_digest=raw["runner_source_digest"],
            task_plan=plan,
            execution_precommit_digest=raw["execution_precommit_digest"],
            semantic_artifact=semantic,
            rubric_specs=specs,  # type: ignore[arg-type]
            side_0_support_by_rank=tuple(
                tuple(ObjectBongardRubricObserverArtifact.from_data(item) for item in block)
                for block in raw["side_0_support_by_rank"]
            ),  # type: ignore[arg-type]
            side_1_support_by_rank=tuple(
                tuple(ObjectBongardRubricObserverArtifact.from_data(item) for item in block)
                for block in raw["side_1_support_by_rank"]
            ),  # type: ignore[arg-type]
            support_digest=raw["support_digest"],
            version_spaces=spaces,  # type: ignore[arg-type]
            slate_selection=selection,
            freeze=(None if raw["freeze"] is None else ObjectBongardRubricTaskFreeze.from_data(raw["freeze"])),
            freeze_commit=(None if raw["freeze_commit"] is None else ObjectBongardRubricTaskFreezeCommit.from_data(raw["freeze_commit"])),
            side_0_query=(None if raw["side_0_query"] is None else ObjectBongardRubricObserverArtifact.from_data(raw["side_0_query"])),
            side_1_query=(None if raw["side_1_query"] is None else ObjectBongardRubricObserverArtifact.from_data(raw["side_1_query"])),
            query_results=tuple(
                ObjectBongardRubricTaskQueryResult.from_data(item)
                for item in raw["query_results"]
            ),
            correct_count=raw["correct_count"],
            abstention_count=raw["abstention_count"],
            score_denominator=raw["score_denominator"],
            accuracy_ppm=raw["accuracy_ppm"],
            selection_replay_calls_made=raw["selection_replay_calls_made"],
            freeze_commit_calls_made=raw["freeze_commit_calls_made"],
            freeze_reload_calls_made=raw["freeze_reload_calls_made"],
            query_source_calls_made=raw["query_source_calls_made"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricTaskRunnerError("two-rank task archive is not canonical")
        return result


def _make_archive_v2(**values: object) -> ObjectBongardRubricTaskRunArchive:
    provisional = object.__new__(ObjectBongardRubricTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricTaskRunArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_v2_content(provisional)),
    )


def run_object_bongard_rubric_task(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    side_0_support_by_rank: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
    side_1_support_by_rank: Sequence[Sequence[ObjectBongardRubricObserverArtifact]],
    *,
    execution_precommit_digest: str,
    freeze_committer: FreezeCommitter,
    freeze_reloader: FreezeReloader,
    query_source: QuerySource,
) -> ObjectBongardRubricTaskRunArchive:
    """Select on two exact 6+6 support blocks, freeze, then permit 1+1 query."""

    plan, semantic, specs, precommit = _canonical_parents(
        task_plan, semantic_artifact, execution_precommit_digest
    )
    positives, negatives = _canonical_support(
        plan, specs, side_0_support_by_rank, side_1_support_by_rank
    )
    spaces = tuple(
        cold_verify_object_bongard_rubric_support_version_space(
            built, specs[rank], positives[rank], negatives[rank]
        )
        for rank in (0, 1)
        for built in (
            build_object_bongard_rubric_support_version_space(
                specs[rank], positives[rank], negatives[rank]
            ),
        )
    )
    selection = select_object_bongard_rubric_slate(specs, spaces)
    selection = cold_verify_object_bongard_rubric_slate(selection, specs, spaces)
    support_digest = _support_digest(specs, positives, negatives)
    common: dict[str, object] = {
        "runner_source_digest": object_bongard_rubric_task_runner_source_digest(),
        "task_plan": plan,
        "execution_precommit_digest": precommit,
        "semantic_artifact": semantic,
        "rubric_specs": specs,
        "side_0_support_by_rank": positives,
        "side_1_support_by_rank": negatives,
        "support_digest": support_digest,
        "version_spaces": spaces,
        "slate_selection": selection,
        "selection_replay_calls_made": 1,
    }
    selected = selection.selected_candidate
    if selected is None:
        return _make_archive_v2(
            status=_slate_gap_status(spaces),
            **common,
            freeze=None,
            freeze_commit=None,
            side_0_query=None,
            side_1_query=None,
            query_results=(),
            correct_count=0,
            abstention_count=0,
            score_denominator=0,
            accuracy_ppm=None,
            freeze_commit_calls_made=0,
            freeze_reload_calls_made=0,
            query_source_calls_made=0,
        )
    selected_spec = next(
        item for item in specs if item.spec_digest == selected.rubric_spec_digest
    )
    freeze = ObjectBongardRubricTaskFreeze.seal(
        task_plan=plan,
        execution_precommit_digest=precommit,
        semantic_artifact=semantic,
        rubric_specs=specs,
        support_digest=support_digest,
        slate_selection=selection,
    )
    freeze_data = ObjectBongardRubricTaskFreeze.from_data(freeze.to_data()).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, ObjectBongardRubricTaskFreezeCommit)
        else ObjectBongardRubricTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)

    # This mandatory durable reload is the last operation before query access.
    reloaded = freeze_reloader(commit.to_data())
    if reloaded != freeze_bytes:
        raise ObjectBongardRubricTaskRunnerError("durable two-rank freeze reload differs")
    try:
        decoded_reload = json.loads(reloaded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricTaskRunnerError("durable freeze reload is not JSON") from exc
    if ObjectBongardRubricTaskFreeze.from_data(decoded_reload) != freeze:
        raise ObjectBongardRubricTaskRunnerError("durable two-rank freeze object differs")

    raw_queries = query_source(freeze_data, commit.to_data())
    if not isinstance(raw_queries, Mapping) or set(raw_queries) != {"side_0", "side_1"}:
        raise ObjectBongardRubricTaskRunnerError(
            "query source must return exactly side_0 and side_1"
        )
    queries = tuple(_canonical_artifact(raw_queries[side]) for side in ("side_0", "side_1"))
    if (
        tuple(item.panel_id for item in queries)
        != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
        or any(item.rubric_spec != selected_spec for item in queries)
        or any(
            item.catalog_digest != positives[0][0].catalog_digest
            or item.runtime_identity_digest != positives[0][0].runtime_identity_digest
            for item in queries
        )
    ):
        raise ObjectBongardRubricTaskRunnerError(
            "query artifacts differ from the frozen selected spec or runtime"
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
    return _make_archive_v2(
        status=ObjectBongardRubricTaskRunStatus.COMPLETE,
        **common,
        freeze=freeze,
        freeze_commit=commit,
        side_0_query=queries[0],
        side_1_query=queries[1],
        query_results=results,
        correct_count=correct,
        abstention_count=sum(item.abstained for item in results),
        score_denominator=2,
        accuracy_ppm=correct * 500_000,
        freeze_commit_calls_made=1,
        freeze_reload_calls_made=1,
        query_source_calls_made=1,
    )


def cold_replay_object_bongard_rubric_task(
    archive: ObjectBongardRubricTaskRunArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> ObjectBongardRubricTaskRunArchive:
    """Replay both support version spaces and selection with zero model calls."""

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
    restored = ObjectBongardRubricTaskRunArchive.from_data(
        archive.to_data() if isinstance(archive, ObjectBongardRubricTaskRunArchive) else archive
    )
    if restored.record_digest != expected:
        raise ObjectBongardRubricTaskRunnerError("cold rubric task archive digest differs")
    return restored


__all__ = (
    "ARCHIVE_SCHEMA", "FREEZE_COMMIT_SCHEMA", "FREEZE_SCHEMA",
    "QUERY_RESULT_SCHEMA", "RUNNER_ID", "ObjectBongardRubricTaskFreeze",
    "ObjectBongardRubricTaskFreezeCommit", "ObjectBongardRubricTaskQueryResult",
    "ObjectBongardRubricTaskRunArchive", "ObjectBongardRubricTaskRunStatus",
    "ObjectBongardRubricTaskRunnerError", "cold_replay_object_bongard_rubric_task",
    "object_bongard_rubric_task_runner_source_digest", "run_object_bongard_rubric_task",
)
