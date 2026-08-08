"""Durable Python runner for the two-rank whole-panel rubric path.

The runner consumes two already sealed ranks of six-plus-six whole-panel
observations.  It builds the two singleton Python version spaces, selects the
first bounded-admissible rank, commits and reloads the exact freeze bytes, and
only then asks a callback for the two query observations.  Query abstentions
and observer errors are incorrect under the fixed denominator, while coverage
is reported separately.

No query pixels enter support selection.  No model ranks candidates, and no
atlas observer, ranker, or Lean component participates in identity, decision,
scoring, or cold replay.
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
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
)
from bongard.object_bongard_panel_rubric_slate import (
    ObjectBongardPanelRubricSlateSelection,
    cold_verify_object_bongard_panel_rubric_slate,
    object_bongard_panel_rubric_slate_algorithm_digest,
    select_object_bongard_panel_rubric_slate,
)
from bongard.object_bongard_panel_rubric_version_space import (
    ObjectBongardPanelRubricCandidate,
    ObjectBongardPanelRubricCandidateEvaluation,
    ObjectBongardPanelRubricSupportVersionSpace,
    PanelRubricPredicateOperator,
    PanelRubricSupportGapKind,
    build_object_bongard_panel_rubric_support_version_space,
    cold_verify_object_bongard_panel_rubric_support_version_space,
    evaluate_object_bongard_panel_rubric_candidate,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUNNER_ID = "bongard.panel-rubric-task/two-rank-bounded-freeze-query-v1"
FREEZE_SCHEMA = "gkm.bongard-panel-rubric-task-freeze.v1"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-panel-rubric-task-freeze-commit.v1"
QUERY_RESULT_SCHEMA = "gkm.bongard-panel-rubric-task-query-result.v1"
ARCHIVE_SCHEMA = "gkm.bongard-panel-rubric-task-run-archive.v1"
PANEL_RUBRIC_TASK_SCORE_DENOMINATOR = 2

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardPanelRubricTaskRunnerError(RuntimeError):
    """A parent, support, freeze, query, or replay boundary failed closed."""


class ObjectBongardPanelRubricTaskRunStatus(str, Enum):
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
        "lean_removal_changes_decision": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "disjunction_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "threshold_tuning_allowed": False,
        "ordinal_sums_allowed": False,
        "retries_allowed": False,
        "model_selects_candidate": False,
    }


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardPanelRubricTaskRunnerError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardPanelRubricTaskRunnerError(
            f"{label} must be a sha256: address"
        )
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardPanelRubricTaskRunnerError(
            f"{label} must be a bounded identifier"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardPanelRubricTaskRunnerError(
            f"{label} fields differ from schema"
        )
    return value


def object_bongard_panel_rubric_task_runner_source_digest() -> str:
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
        raise ObjectBongardPanelRubricTaskRunnerError(
            "semantic artifact does not bind the exact task plan and precommit"
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
        tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != 2
        or any(item.semantic_artifact_digest != semantic.artifact_digest for item in specs)
    ):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "semantic artifact does not yield two distinct ranked rubric specs"
        )
    return plan, semantic, specs, precommit  # type: ignore[return-value]


def _canonical_artifact(
    value: ObjectBongardPanelRubricArtifact,
) -> ObjectBongardPanelRubricArtifact:
    if not isinstance(value, ObjectBongardPanelRubricArtifact):
        raise TypeError(
            "panel evidence must contain ObjectBongardPanelRubricArtifact"
        )
    restored = ObjectBongardPanelRubricArtifact.from_data(value.to_data())
    if restored != value:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "panel observer artifact cold round trip differs"
        )
    return restored


def _canonical_support(
    plan: ObjectBongardTaskPlan,
    specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
    precommit: str,
    side_0: Sequence[Sequence[ObjectBongardPanelRubricArtifact]],
    side_1: Sequence[Sequence[ObjectBongardPanelRubricArtifact]],
) -> tuple[
    tuple[
        tuple[ObjectBongardPanelRubricArtifact, ...],
        tuple[ObjectBongardPanelRubricArtifact, ...],
    ],
    tuple[
        tuple[ObjectBongardPanelRubricArtifact, ...],
        tuple[ObjectBongardPanelRubricArtifact, ...],
    ],
]:
    if isinstance(side_0, (str, bytes)) or isinstance(side_1, (str, bytes)):
        raise TypeError("ranked panel support must be a sequence of blocks")
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
    if len(positives) != 2 or len(negatives) != 2:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "support evidence must contain exactly two ranked blocks per side"
        )
    for blocks, expected_ids in (
        (positives, plan.side_0_support_panel_ids),
        (negatives, plan.side_1_support_panel_ids),
    ):
        if any(
            len(block) != 6
            or tuple(item.panel_id for item in block) != expected_ids
            for block in blocks
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "each rank must bind the exact sorted six-plus-six support IDs"
            )
    all_artifacts = tuple(
        item
        for blocks in (positives, negatives)
        for block in blocks
        for item in block
    )
    if (
        len(all_artifacts) != 24
        or any(
            item.rubric_spec != specs[rank]
            for rank in (0, 1)
            for block in (positives[rank], negatives[rank])
            for item in block
        )
        or any(item.observation_context_digest != precommit for item in all_artifacts)
        or len({item.protocol_digest for item in all_artifacts}) != 1
        or len({item.runtime_identity_digest for item in all_artifacts}) != 1
    ):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "ranked support differs in spec, precommit, protocol, or runtime"
        )
    return positives, negatives  # type: ignore[return-value]


def _support_digest(
    specs: Sequence[ObjectBongardRubricSpec],
    positives: Sequence[Sequence[ObjectBongardPanelRubricArtifact]],
    negatives: Sequence[Sequence[ObjectBongardPanelRubricArtifact]],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-task-support.v1",
            "rubric_specs": [item.to_data() for item in specs],
            "side_0_positive_artifacts_by_rank": [
                [item.to_data() for item in block] for block in positives
            ],
            "side_1_negative_artifacts_by_rank": [
                [item.to_data() for item in block] for block in negatives
            ],
            "candidate_rank_order": [0, 1],
            "support_panels_per_side_per_spec": 6,
            "whole_panel_observations_per_task": 24,
            "support_labels_supplied_to_python_only": True,
            "query_material_included": False,
            **_authority_data(),
        }
    )


def _query_result_content(
    value: "ObjectBongardPanelRubricTaskQueryResult",
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
class ObjectBongardPanelRubricTaskQueryResult:
    side: str
    panel_id: str
    expected_disposition: Disposition
    evaluation: ObjectBongardPanelRubricCandidateEvaluation
    correct: bool
    incorrect: bool
    covered: bool
    abstained: bool
    result_digest: str

    def __post_init__(self) -> None:
        if self.side not in ("side_0", "side_1"):
            raise ObjectBongardPanelRubricTaskRunnerError("query side is unknown")
        expected = (
            Disposition.PRESENT
            if self.side == "side_0"
            else Disposition.CERTIFIED_ABSENT
        )
        if not isinstance(
            self.evaluation, ObjectBongardPanelRubricCandidateEvaluation
        ):
            raise TypeError("query evaluation has the wrong type")
        definite = self.evaluation.disposition in (
            Disposition.PRESENT,
            Disposition.CERTIFIED_ABSENT,
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
            raise ObjectBongardPanelRubricTaskRunnerError(
                "query result differs from fixed scoring and coverage"
            )

    @classmethod
    def create(
        cls,
        side: str,
        artifact: ObjectBongardPanelRubricArtifact,
        evaluation: ObjectBongardPanelRubricCandidateEvaluation,
    ) -> "ObjectBongardPanelRubricTaskQueryResult":
        expected = (
            Disposition.PRESENT
            if side == "side_0"
            else Disposition.CERTIFIED_ABSENT
        )
        definite = evaluation.disposition in (
            Disposition.PRESENT,
            Disposition.CERTIFIED_ABSENT,
        )
        values: dict[str, object] = {
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
            **values,  # type: ignore[arg-type]
            result_digest=canonical_digest(_query_result_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricTaskQueryResult":
        raw = _fields(
            value,
            {
                "schema", "side", "panel_id", "expected_disposition",
                "evaluation", "correct", "incorrect", "covered", "abstained",
                "fixed_denominator_contribution",
                "indeterminate_or_error_counts_as_incorrect",
                "coverage_requires_definite_disposition", *_authority_data(),
                "result_digest",
            },
            "panel rubric query result",
        )
        if (
            raw["schema"] != QUERY_RESULT_SCHEMA
            or raw["fixed_denominator_contribution"] != 1
            or raw["indeterminate_or_error_counts_as_incorrect"] is not True
            or raw["coverage_requires_definite_disposition"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "panel query result policy differs"
            )
        try:
            expected = Disposition(raw["expected_disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "query expected disposition differs"
            ) from exc
        result = cls(
            raw["side"],
            raw["panel_id"],
            expected,
            ObjectBongardPanelRubricCandidateEvaluation.from_data(
                raw["evaluation"]
            ),
            raw["correct"],
            raw["incorrect"],
            raw["covered"],
            raw["abstained"],
            raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "panel query result is not canonical"
            )
        return result


def _freeze_content(
    value: "ObjectBongardPanelRubricTaskFreeze",
) -> dict[str, object]:
    selection = value.slate_selection
    selected = selection.selected_candidate
    selected_spec = selection.selected_rubric_spec
    if selected is None or selected_spec is None:  # pragma: no cover
        raise ObjectBongardPanelRubricTaskRunnerError(
            "cannot serialize an empty freeze"
        )
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
        # The release-gate protocol predates this deterministic slate.  These
        # aliases bind the exact slate; none is a response from a ranker.
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_response_digest_is_legacy_alias_for_slate_selection": True,
        "selected_rubric_spec": selected_spec.to_data(),
        "selected_rubric_spec_digest": selected_spec.spec_digest,
        "selected_candidate": selected.to_data(),
        "selected_candidate_digest": selected.candidate_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_formula": value.selected_formula,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_pixels_included": False,
        "query_observer_artifacts_included": False,
        "selected_spec_candidate_formula_frozen_before_query_source": True,
        "candidate_order": ["rank-0/panel", "rank-1/panel"],
        "strict_exact_six_plus_six_is_diagnostic_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricTaskFreeze:
    """Exact selected spec/candidate/formula committed before query access."""

    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    support_digest: str
    observer_protocol_digest: str
    observer_runtime_identity_digest: str
    slate_algorithm_digest: str
    version_space_digests: tuple[str, str]
    slate_selection: ObjectBongardPanelRubricSlateSelection
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
            "observer_protocol_digest", "observer_runtime_identity_digest",
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
            raise ObjectBongardPanelRubricTaskRunnerError(
                "freeze requires two rubric specs"
            )
        specs = tuple(
            ObjectBongardRubricSpec.from_data(item.to_data())
            for item in self.rubric_specs
        )
        if not isinstance(
            self.slate_selection, ObjectBongardPanelRubricSlateSelection
        ):
            raise TypeError("freeze slate selection has the wrong type")
        selection = ObjectBongardPanelRubricSlateSelection.from_data(
            self.slate_selection.to_data()
        )
        selected = selection.selected_candidate
        selected_spec = selection.selected_rubric_spec
        spaces = selection.version_spaces
        if (
            selected is None
            or selected_spec is None
            or specs != self.rubric_specs
            or tuple(item.candidate_rank for item in specs) != (0, 1)
            or self.runner_source_digest
            != object_bongard_panel_rubric_task_runner_source_digest()
            or selection != self.slate_selection
            or selection.rubric_specs != specs
            or self.semantic_artifact_digest != selection.semantic_artifact_digest
            or self.slate_algorithm_digest
            != object_bongard_panel_rubric_slate_algorithm_digest()
            or self.slate_algorithm_digest != selection.algorithm_digest
            or self.version_space_digests
            != tuple(item.version_space_digest for item in spaces)
            or self.observer_protocol_digest != spaces[0].observer_protocol_digest
            or self.observer_runtime_identity_digest
            != spaces[0].observer_runtime_identity_digest
            or len({self.version_space_digest, self.support_version_space_digest,
                    self.rank_response_digest, selection.selection_digest}) != 1
            or self.selected_predicate_digest != selected.candidate_digest
            or self.selected_formula != selected.formula
            or selected.operator is not PanelRubricPredicateOperator.AT_LEAST
            or self.sealed_query_panel_ids
            != tuple(self.sealed_query_panel_ids)
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(_IDENTIFIER.fullmatch(item) is None for item in self.sealed_query_panel_ids)
            or self.record_digest != _content_address(_freeze_content(self))
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel task freeze content differs"
            )

    @property
    def selected_candidate(self) -> ObjectBongardPanelRubricCandidate:
        selected = self.slate_selection.selected_candidate
        if selected is None:  # pragma: no cover
            raise ObjectBongardPanelRubricTaskRunnerError("freeze selection is empty")
        return selected

    @property
    def selected_rubric_spec(self) -> ObjectBongardRubricSpec:
        selected = self.slate_selection.selected_rubric_spec
        if selected is None:  # pragma: no cover
            raise ObjectBongardPanelRubricTaskRunnerError("freeze selection is empty")
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
        semantic_artifact: ObjectBongardSemanticArtifact,
        rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
        support_digest: str,
        slate_selection: ObjectBongardPanelRubricSlateSelection,
    ) -> "ObjectBongardPanelRubricTaskFreeze":
        selected = slate_selection.selected_candidate
        if selected is None:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "cannot freeze a slate without a bounded survivor"
            )
        spaces = slate_selection.version_spaces
        values: dict[str, object] = {
            "runner_source_digest": (
                object_bongard_panel_rubric_task_runner_source_digest()
            ),
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.record_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "semantic_artifact_digest": semantic_artifact.artifact_digest,
            "rubric_specs": rubric_specs,
            "support_digest": support_digest,
            "observer_protocol_digest": spaces[0].observer_protocol_digest,
            "observer_runtime_identity_digest": (
                spaces[0].observer_runtime_identity_digest
            ),
            "slate_algorithm_digest": slate_selection.algorithm_digest,
            "version_space_digests": tuple(
                item.version_space_digest for item in spaces
            ),
            "slate_selection": slate_selection,
            "version_space_digest": slate_selection.selection_digest,
            "support_version_space_digest": slate_selection.selection_digest,
            "rank_response_digest": slate_selection.selection_digest,
            "selected_predicate_digest": selected.candidate_digest,
            "selected_formula": selected.formula,
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
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact_digest", "rubric_specs", "rubric_spec_digests",
                "support_digest", "observer_protocol_digest",
                "observer_runtime_identity_digest", "slate_algorithm_digest",
                "version_space_digests", "slate_selection",
                "slate_selection_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_rubric_spec", "selected_rubric_spec_digest",
                "selected_candidate", "selected_candidate_digest",
                "selected_predicate_digest", "selected_formula",
                "sealed_query_panel_ids", "query_pixels_included",
                "query_observer_artifacts_included",
                "selected_spec_candidate_formula_frozen_before_query_source",
                "candidate_order", "strict_exact_six_plus_six_is_diagnostic_only",
                *_authority_data(), "record_digest",
            },
            "whole-panel task freeze",
        )
        for name in (
            "rubric_specs", "rubric_spec_digests", "version_space_digests",
            "sealed_query_panel_ids", "candidate_order",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardPanelRubricTaskRunnerError(
                    f"freeze {name} must be a JSON list"
                )
        specs = tuple(
            ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"]
        )
        selection = ObjectBongardPanelRubricSlateSelection.from_data(
            raw["slate_selection"]
        )
        selected = selection.selected_candidate
        selected_spec = selection.selected_rubric_spec
        if (
            raw["schema"] != FREEZE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["slate_selection_digest"] != selection.selection_digest
            or selected is None
            or selected_spec is None
            or raw["selected_rubric_spec"] != selected_spec.to_data()
            or raw["selected_rubric_spec_digest"] != selected_spec.spec_digest
            or raw["selected_candidate"] != selected.to_data()
            or raw["selected_candidate_digest"] != selected.candidate_digest
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["query_pixels_included"] is not False
            or raw["query_observer_artifacts_included"] is not False
            or raw["selected_spec_candidate_formula_frozen_before_query_source"] is not True
            or raw["candidate_order"] != ["rank-0/panel", "rank-1/panel"]
            or raw["strict_exact_six_plus_six_is_diagnostic_only"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze policy differs"
            )
        result = cls(
            runner_source_digest=raw["runner_source_digest"],
            task_id=raw["task_id"],
            task_plan_digest=raw["task_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            semantic_artifact_digest=raw["semantic_artifact_digest"],
            rubric_specs=specs,  # type: ignore[arg-type]
            support_digest=raw["support_digest"],
            observer_protocol_digest=raw["observer_protocol_digest"],
            observer_runtime_identity_digest=raw[
                "observer_runtime_identity_digest"
            ],
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
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze is not canonical"
            )
        return result


def _commit_content(
    value: "ObjectBongardPanelRubricTaskFreezeCommit",
) -> dict[str, object]:
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
        "task_freeze_store_receipt_digest": (
            value.task_freeze_store_receipt_digest
        ),
        "durably_persisted_before_query_source": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricTaskFreezeCommit:
    """Receipt for the exact canonical freeze byte string."""

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
        _address(
            self.execution_precommit_digest,
            "commit execution precommit digest",
        )
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
            len(
                {
                    self.slate_selection_digest,
                    self.version_space_digest,
                    self.support_version_space_digest,
                    self.rank_response_digest,
                }
            )
            != 1
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze commit differs"
            )

    @classmethod
    def seal(
        cls,
        freeze: ObjectBongardPanelRubricTaskFreeze,
        exact_freeze_payload: bytes,
        *,
        task_freeze_store_receipt_digest: str,
    ) -> "ObjectBongardPanelRubricTaskFreezeCommit":
        if not isinstance(freeze, ObjectBongardPanelRubricTaskFreeze):
            raise TypeError("freeze must be ObjectBongardPanelRubricTaskFreeze")
        expected = canonical_json(freeze.to_data()) + b"\n"
        if exact_freeze_payload != expected:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "freeze payload bytes are not exact canonical JSON"
            )
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
            "exact_freeze_payload_digest": (
                "sha256:" + hashlib.sha256(expected).hexdigest()
            ),
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
        freeze: ObjectBongardPanelRubricTaskFreeze,
        exact_freeze_payload: bytes,
    ) -> None:
        replayed = type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=(
                self.task_freeze_store_receipt_digest
            ),
        )
        if self != replayed:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze commit replay differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "task_id", "task_plan_digest",
                "execution_precommit_digest", "slate_selection_digest",
                "version_space_digest", "support_version_space_digest",
                "rank_response_digest",
                "rank_response_digest_is_legacy_alias_for_slate_selection",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest",
                "task_freeze_store_receipt_digest",
                "durably_persisted_before_query_source", *_authority_data(),
                "record_digest",
            },
            "whole-panel task freeze commit",
        )
        if (
            raw["schema"] != FREEZE_COMMIT_SCHEMA
            or raw["rank_response_digest_is_legacy_alias_for_slate_selection"] is not True
            or raw["durably_persisted_before_query_source"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze commit policy differs"
            )
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
            task_freeze_store_receipt_digest=raw[
                "task_freeze_store_receipt_digest"
            ],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel freeze commit is not canonical"
            )
        return result


FreezeCommitter = Callable[
    [bytes], ObjectBongardPanelRubricTaskFreezeCommit | Mapping[str, Any]
]
FreezeReloader = Callable[[Mapping[str, Any]], bytes]
QuerySource = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, ObjectBongardPanelRubricArtifact],
]


def _gap_status(
    spaces: Sequence[ObjectBongardPanelRubricSupportVersionSpace],
) -> ObjectBongardPanelRubricTaskRunStatus:
    if any(item.survivor_candidate_digests for item in spaces):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "empty slate cannot contain a support survivor"
        )
    if any(item.gap is None for item in spaces):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "empty slate version space lacks a typed gap"
        )
    return (
        ObjectBongardPanelRubricTaskRunStatus.WITNESS_GAP
        if any(
            item.gap.kind is PanelRubricSupportGapKind.WITNESS_GAP
            for item in spaces
            if item.gap is not None
        )
        else ObjectBongardPanelRubricTaskRunStatus.LANGUAGE_GAP
    )


def _archive_content(
    value: "ObjectBongardPanelRubricTaskRunArchive",
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
            [item.to_data() for item in block]
            for block in value.side_0_support_by_rank
        ],
        "side_1_support_by_rank": [
            [item.to_data() for item in block]
            for block in value.side_1_support_by_rank
        ],
        "support_digest": value.support_digest,
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "version_space_digests": [
            item.version_space_digest for item in value.version_spaces
        ],
        "slate_selection": value.slate_selection.to_data(),
        "slate_selection_digest": value.slate_selection.selection_digest,
        "selection_model_calls_made": 0,
        "selection_replay_calls_made": value.selection_replay_calls_made,
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
        "whole_panel_support_observations": 24,
        "query_observations_per_task": 2,
        "candidate_order": ["rank-0/panel", "rank-1/panel"],
        "rank_zero_selected_when_bounded_admissible": True,
        "strict_exact_six_plus_six_is_diagnostic_only": True,
        "query_uses_frozen_selected_python_candidate_only": True,
        "gap_counts_as_two_uncovered_incorrect_abstentions": True,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricTaskRunArchive:
    """Self-contained artifact for support selection, freeze, and query score."""

    status: ObjectBongardPanelRubricTaskRunStatus
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    semantic_artifact: ObjectBongardSemanticArtifact
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    side_0_support_by_rank: tuple[
        tuple[ObjectBongardPanelRubricArtifact, ...],
        tuple[ObjectBongardPanelRubricArtifact, ...],
    ]
    side_1_support_by_rank: tuple[
        tuple[ObjectBongardPanelRubricArtifact, ...],
        tuple[ObjectBongardPanelRubricArtifact, ...],
    ]
    support_digest: str
    version_spaces: tuple[
        ObjectBongardPanelRubricSupportVersionSpace,
        ObjectBongardPanelRubricSupportVersionSpace,
    ]
    slate_selection: ObjectBongardPanelRubricSlateSelection
    freeze: ObjectBongardPanelRubricTaskFreeze | None
    freeze_commit: ObjectBongardPanelRubricTaskFreezeCommit | None
    side_0_query: ObjectBongardPanelRubricArtifact | None
    side_1_query: ObjectBongardPanelRubricArtifact | None
    query_results: tuple[ObjectBongardPanelRubricTaskQueryResult, ...]
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
        if not isinstance(self.status, ObjectBongardPanelRubricTaskRunStatus):
            raise TypeError("panel rubric task status has the wrong type")
        if (
            self.runner_source_digest
            != object_bongard_panel_rubric_task_runner_source_digest()
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive runner source differs"
            )
        plan, semantic, specs, precommit = _canonical_parents(
            self.task_plan,
            self.semantic_artifact,
            self.execution_precommit_digest,
        )
        positives, negatives = _canonical_support(
            plan,
            specs,
            precommit,
            self.side_0_support_by_rank,
            self.side_1_support_by_rank,
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
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive parent or support inventory differs"
            )
        replayed_spaces = tuple(
            cold_verify_object_bongard_panel_rubric_support_version_space(
                self.version_spaces[rank],
                specs[rank],
                positives[rank],
                negatives[rank],
            )
            for rank in (0, 1)
        )
        if replayed_spaces != self.version_spaces:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive support version-space replay differs"
            )
        selection = cold_verify_object_bongard_panel_rubric_slate(
            self.slate_selection,
            specs,
            replayed_spaces,
        )
        if (
            selection != self.slate_selection
            or self.selection_replay_calls_made != 1
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive slate replay differs"
            )
        metric_names = (
            "correct_count", "incorrect_count", "abstention_count",
            "coverage_count", "score_denominator", "accuracy_ppm",
            "coverage_ppm", "selection_replay_calls_made",
            "freeze_commit_calls_made", "freeze_reload_calls_made",
            "query_source_calls_made",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in metric_names
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive counters must be nonnegative integers"
            )
        if self.score_denominator != PANEL_RUBRIC_TASK_SCORE_DENOMINATOR:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "every task must retain the fixed query denominator of two"
            )
        if selection.selected_candidate is None:
            self._validate_gap(replayed_spaces)
        else:
            self._validate_complete(
                plan=plan,
                semantic=semantic,
                specs=specs,
                precommit=precommit,
                positives=positives,
                selection=selection,
            )
        _raw_digest(self.record_digest, "archive digest")
        if self.record_digest != canonical_digest(_archive_content(self)):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel task archive digest differs"
            )

    @property
    def selected_candidate(self) -> ObjectBongardPanelRubricCandidate | None:
        return self.slate_selection.selected_candidate

    @property
    def selected_rubric_spec(self) -> ObjectBongardRubricSpec | None:
        return self.slate_selection.selected_rubric_spec

    def _validate_gap(
        self,
        spaces: Sequence[ObjectBongardPanelRubricSupportVersionSpace],
    ) -> None:
        expected_status = _gap_status(spaces)
        if (
            self.status is not expected_status
            or self.freeze is not None
            or self.freeze_commit is not None
            or self.side_0_query is not None
            or self.side_1_query is not None
            or self.query_results
            or self.correct_count != 0
            or self.incorrect_count != 2
            or self.abstention_count != 2
            or self.coverage_count != 0
            or self.accuracy_ppm != 0
            or self.coverage_ppm != 0
            or self.freeze_commit_calls_made != 0
            or self.freeze_reload_calls_made != 0
            or self.query_source_calls_made != 0
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "typed support gap crossed a later phase or escaped scoring"
            )

    def _validate_complete(
        self,
        *,
        plan: ObjectBongardTaskPlan,
        semantic: ObjectBongardSemanticArtifact,
        specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
        precommit: str,
        positives: tuple[
            tuple[ObjectBongardPanelRubricArtifact, ...],
            tuple[ObjectBongardPanelRubricArtifact, ...],
        ],
        selection: ObjectBongardPanelRubricSlateSelection,
    ) -> None:
        selected = selection.selected_candidate
        selected_spec = selection.selected_rubric_spec
        if (
            selected is None
            or selected_spec is None
            or self.status is not ObjectBongardPanelRubricTaskRunStatus.COMPLETE
            or self.freeze is None
            or self.freeze_commit is None
            or self.side_0_query is None
            or self.side_1_query is None
            or len(self.query_results) != 2
            or self.freeze_commit_calls_made != 1
            or self.freeze_reload_calls_made != 1
            or self.query_source_calls_made != 1
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "complete archive phase inventory differs"
            )
        expected_freeze = ObjectBongardPanelRubricTaskFreeze.seal(
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
        protocol = positives[0][0].protocol_digest
        runtime = positives[0][0].runtime_identity_digest
        if (
            self.freeze != expected_freeze
            or tuple(item.panel_id for item in queries)
            != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
            or any(item.rubric_spec != selected_spec for item in queries)
            or any(item.observation_context_digest != precommit for item in queries)
            or any(item.protocol_digest != protocol for item in queries)
            or any(item.runtime_identity_digest != runtime for item in queries)
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "query artifacts differ from the frozen spec or observer runtime"
            )
        evaluations = tuple(
            evaluate_object_bongard_panel_rubric_candidate(selected, artifact)
            for artifact in queries
        )
        expected_results = tuple(
            ObjectBongardPanelRubricTaskQueryResult.create(
                side, artifact, evaluation
            )
            for side, artifact, evaluation in zip(
                ("side_0", "side_1"),
                queries,
                evaluations,
                strict=True,
            )
        )
        correct = sum(item.correct for item in expected_results)
        incorrect = sum(item.incorrect for item in expected_results)
        abstained = sum(item.abstained for item in expected_results)
        covered = sum(item.covered for item in expected_results)
        if (
            self.query_results != expected_results
            or self.correct_count != correct
            or self.incorrect_count != incorrect
            or self.abstention_count != abstained
            or self.coverage_count != covered
            or self.accuracy_ppm != correct * 500_000
            or self.coverage_ppm != covered * 500_000
            or correct + incorrect != 2
            or covered + abstained != 2
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive fixed-denominator score or coverage differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "status", "runner_source_digest",
                "task_plan", "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact", "semantic_artifact_digest", "rubric_specs",
                "rubric_spec_digests", "side_0_support_by_rank",
                "side_1_support_by_rank", "support_digest", "version_spaces",
                "version_space_digests", "slate_selection",
                "slate_selection_digest", "selection_model_calls_made",
                "selection_replay_calls_made", "freeze", "freeze_commit",
                "side_0_query", "side_1_query", "query_results",
                "correct_count", "incorrect_count", "abstention_count",
                "coverage_count", "score_denominator", "accuracy_ppm",
                "coverage_ppm", "freeze_commit_calls_made",
                "freeze_reload_calls_made", "query_source_calls_made",
                "support_spec_count", "support_observations_per_spec",
                "whole_panel_support_observations",
                "query_observations_per_task", "candidate_order",
                "rank_zero_selected_when_bounded_admissible",
                "strict_exact_six_plus_six_is_diagnostic_only",
                "query_uses_frozen_selected_python_candidate_only",
                "gap_counts_as_two_uncovered_incorrect_abstentions",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "whole-panel task archive",
        )
        for name in (
            "rubric_specs", "rubric_spec_digests", "side_0_support_by_rank",
            "side_1_support_by_rank", "version_spaces",
            "version_space_digests", "query_results", "candidate_order",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardPanelRubricTaskRunnerError(
                    f"archive {name} must be a JSON list"
                )
        if any(
            not isinstance(block, list)
            for name in ("side_0_support_by_rank", "side_1_support_by_rank")
            for block in raw[name]
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "archive ranked support blocks must be JSON lists"
            )
        plan = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        semantic = ObjectBongardSemanticArtifact.from_data(
            raw["semantic_artifact"]
        )
        specs = tuple(
            ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"]
        )
        spaces = tuple(
            ObjectBongardPanelRubricSupportVersionSpace.from_data(item)
            for item in raw["version_spaces"]
        )
        selection = ObjectBongardPanelRubricSlateSelection.from_data(
            raw["slate_selection"]
        )
        if (
            raw["schema"] != ARCHIVE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["task_plan_digest"] != plan.record_digest
            or raw["semantic_artifact_digest"] != semantic.artifact_digest
            or raw["rubric_spec_digests"] != [item.spec_digest for item in specs]
            or raw["version_space_digests"]
            != [item.version_space_digest for item in spaces]
            or raw["slate_selection_digest"] != selection.selection_digest
            or raw["selection_model_calls_made"] != 0
            or raw["support_spec_count"] != 2
            or raw["support_observations_per_spec"] != 12
            or raw["whole_panel_support_observations"] != 24
            or raw["query_observations_per_task"] != 2
            or raw["candidate_order"] != ["rank-0/panel", "rank-1/panel"]
            or raw["rank_zero_selected_when_bounded_admissible"] is not True
            or raw["strict_exact_six_plus_six_is_diagnostic_only"] is not True
            or raw["query_uses_frozen_selected_python_candidate_only"] is not True
            or raw["gap_counts_as_two_uncovered_incorrect_abstentions"] is not True
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel archive policy differs"
            )
        try:
            status = ObjectBongardPanelRubricTaskRunStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel archive status is unknown"
            ) from exc
        result = cls(
            status=status,
            runner_source_digest=raw["runner_source_digest"],
            task_plan=plan,
            execution_precommit_digest=raw["execution_precommit_digest"],
            semantic_artifact=semantic,
            rubric_specs=specs,  # type: ignore[arg-type]
            side_0_support_by_rank=tuple(
                tuple(ObjectBongardPanelRubricArtifact.from_data(item) for item in block)
                for block in raw["side_0_support_by_rank"]
            ),  # type: ignore[arg-type]
            side_1_support_by_rank=tuple(
                tuple(ObjectBongardPanelRubricArtifact.from_data(item) for item in block)
                for block in raw["side_1_support_by_rank"]
            ),  # type: ignore[arg-type]
            support_digest=raw["support_digest"],
            version_spaces=spaces,  # type: ignore[arg-type]
            slate_selection=selection,
            freeze=(
                None
                if raw["freeze"] is None
                else ObjectBongardPanelRubricTaskFreeze.from_data(raw["freeze"])
            ),
            freeze_commit=(
                None
                if raw["freeze_commit"] is None
                else ObjectBongardPanelRubricTaskFreezeCommit.from_data(
                    raw["freeze_commit"]
                )
            ),
            side_0_query=(
                None
                if raw["side_0_query"] is None
                else ObjectBongardPanelRubricArtifact.from_data(raw["side_0_query"])
            ),
            side_1_query=(
                None
                if raw["side_1_query"] is None
                else ObjectBongardPanelRubricArtifact.from_data(raw["side_1_query"])
            ),
            query_results=tuple(
                ObjectBongardPanelRubricTaskQueryResult.from_data(item)
                for item in raw["query_results"]
            ),
            correct_count=raw["correct_count"],
            incorrect_count=raw["incorrect_count"],
            abstention_count=raw["abstention_count"],
            coverage_count=raw["coverage_count"],
            score_denominator=raw["score_denominator"],
            accuracy_ppm=raw["accuracy_ppm"],
            coverage_ppm=raw["coverage_ppm"],
            selection_replay_calls_made=raw["selection_replay_calls_made"],
            freeze_commit_calls_made=raw["freeze_commit_calls_made"],
            freeze_reload_calls_made=raw["freeze_reload_calls_made"],
            query_source_calls_made=raw["query_source_calls_made"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricTaskRunnerError(
                "whole-panel archive is not canonical"
            )
        return result


def _make_archive(**values: object) -> ObjectBongardPanelRubricTaskRunArchive:
    provisional = object.__new__(ObjectBongardPanelRubricTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricTaskRunArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_content(provisional)),
    )


def run_object_bongard_panel_rubric_task(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    side_0_support_by_rank: Sequence[
        Sequence[ObjectBongardPanelRubricArtifact]
    ],
    side_1_support_by_rank: Sequence[
        Sequence[ObjectBongardPanelRubricArtifact]
    ],
    *,
    execution_precommit_digest: str,
    freeze_committer: FreezeCommitter,
    freeze_reloader: FreezeReloader,
    query_source: QuerySource,
) -> ObjectBongardPanelRubricTaskRunArchive:
    """Select from two 6+6 ranks, durably freeze, then score 1+1 queries."""

    plan, semantic, specs, precommit = _canonical_parents(
        task_plan,
        semantic_artifact,
        execution_precommit_digest,
    )
    positives, negatives = _canonical_support(
        plan,
        specs,
        precommit,
        side_0_support_by_rank,
        side_1_support_by_rank,
    )
    spaces = tuple(
        cold_verify_object_bongard_panel_rubric_support_version_space(
            built,
            specs[rank],
            positives[rank],
            negatives[rank],
        )
        for rank in (0, 1)
        for built in (
            build_object_bongard_panel_rubric_support_version_space(
                specs[rank],
                positives[rank],
                negatives[rank],
            ),
        )
    )
    selection = select_object_bongard_panel_rubric_slate(specs, spaces)
    selection = cold_verify_object_bongard_panel_rubric_slate(
        selection,
        specs,
        spaces,
    )
    support_digest = _support_digest(specs, positives, negatives)
    common: dict[str, object] = {
        "runner_source_digest": (
            object_bongard_panel_rubric_task_runner_source_digest()
        ),
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
        "score_denominator": PANEL_RUBRIC_TASK_SCORE_DENOMINATOR,
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
    if selected_spec is None:  # pragma: no cover - selected implies a spec
        raise ObjectBongardPanelRubricTaskRunnerError(
            "selected panel candidate lacks its rubric spec"
        )
    freeze = ObjectBongardPanelRubricTaskFreeze.seal(
        task_plan=plan,
        execution_precommit_digest=precommit,
        semantic_artifact=semantic,
        rubric_specs=specs,
        support_digest=support_digest,
        slate_selection=selection,
    )
    freeze_data = ObjectBongardPanelRubricTaskFreeze.from_data(
        freeze.to_data()
    ).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, ObjectBongardPanelRubricTaskFreezeCommit)
        else ObjectBongardPanelRubricTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)

    # This exact durable reload is deliberately the final operation before the
    # query callback is allowed to create or expose query observations.
    reloaded = freeze_reloader(commit.to_data())
    if not isinstance(reloaded, bytes) or reloaded != freeze_bytes:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "durable whole-panel freeze reload differs"
        )
    try:
        decoded_reload = json.loads(reloaded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "durable freeze reload is not exact JSON"
        ) from exc
    if ObjectBongardPanelRubricTaskFreeze.from_data(decoded_reload) != freeze:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "durable whole-panel freeze object differs"
        )

    raw_queries = query_source(freeze_data, commit.to_data())
    if (
        not isinstance(raw_queries, Mapping)
        or set(raw_queries) != {"side_0", "side_1"}
    ):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "query source must return exactly side_0 and side_1"
        )
    queries = tuple(
        _canonical_artifact(raw_queries[side])
        for side in ("side_0", "side_1")
    )
    protocol = positives[0][0].protocol_digest
    runtime = positives[0][0].runtime_identity_digest
    if (
        tuple(item.panel_id for item in queries)
        != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
        or any(item.rubric_spec != selected_spec for item in queries)
        or any(item.observation_context_digest != precommit for item in queries)
        or any(item.protocol_digest != protocol for item in queries)
        or any(item.runtime_identity_digest != runtime for item in queries)
    ):
        raise ObjectBongardPanelRubricTaskRunnerError(
            "query artifacts differ from the frozen selected spec or runtime"
        )
    evaluations = tuple(
        evaluate_object_bongard_panel_rubric_candidate(selected, artifact)
        for artifact in queries
    )
    results = tuple(
        ObjectBongardPanelRubricTaskQueryResult.create(
            side, artifact, evaluation
        )
        for side, artifact, evaluation in zip(
            ("side_0", "side_1"),
            queries,
            evaluations,
            strict=True,
        )
    )
    correct = sum(item.correct for item in results)
    incorrect = sum(item.incorrect for item in results)
    abstained = sum(item.abstained for item in results)
    covered = sum(item.covered for item in results)
    return _make_archive(
        status=ObjectBongardPanelRubricTaskRunStatus.COMPLETE,
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


def cold_replay_object_bongard_panel_rubric_task(
    archive: ObjectBongardPanelRubricTaskRunArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> ObjectBongardPanelRubricTaskRunArchive:
    """Replay support, selection, freeze, queries, and score without a model."""

    expected = _raw_digest(expected_archive_digest, "expected archive digest")
    supplied = (
        archive.record_digest
        if isinstance(archive, ObjectBongardPanelRubricTaskRunArchive)
        else archive.get("record_digest")
    )
    if supplied != expected:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "panel rubric archive differs from its external commitment"
        )
    restored = ObjectBongardPanelRubricTaskRunArchive.from_data(
        archive.to_data()
        if isinstance(archive, ObjectBongardPanelRubricTaskRunArchive)
        else archive
    )
    if restored.record_digest != expected:
        raise ObjectBongardPanelRubricTaskRunnerError(
            "cold whole-panel task archive digest differs"
        )
    return restored


__all__ = (
    "ARCHIVE_SCHEMA",
    "FREEZE_COMMIT_SCHEMA",
    "FREEZE_SCHEMA",
    "PANEL_RUBRIC_TASK_SCORE_DENOMINATOR",
    "QUERY_RESULT_SCHEMA",
    "RUNNER_ID",
    "FreezeCommitter",
    "FreezeReloader",
    "ObjectBongardPanelRubricTaskFreeze",
    "ObjectBongardPanelRubricTaskFreezeCommit",
    "ObjectBongardPanelRubricTaskQueryResult",
    "ObjectBongardPanelRubricTaskRunArchive",
    "ObjectBongardPanelRubricTaskRunStatus",
    "ObjectBongardPanelRubricTaskRunnerError",
    "QuerySource",
    "cold_replay_object_bongard_panel_rubric_task",
    "object_bongard_panel_rubric_task_runner_source_digest",
    "run_object_bongard_panel_rubric_task",
)
