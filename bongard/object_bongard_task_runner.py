"""Pure-Python freeze/query runner for one generic Bongard object task.

The runner receives candidate-independent scene evidence.  It builds and
cold-verifies the complete finite support version space, lets Codex rank only
the verified survivors, durably freezes the selected immutable Python
predicate, and only then admits the two query evidence records sealed by the
task plan.  Cold replay invokes none of the live callbacks.
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
from bongard.object_bongard_codex_ranker import (
    ObjectBongardRankResponse,
    object_bongard_rank_input_digest,
)
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.prototype_object_profiles import ObjectProfile, ObjectProfileOperator
from bongard.prototype_object_scene_observer import PrototypeSceneObserverStatus
from bongard.prototype_object_version_space import (
    ObjectCandidateSceneEvaluation,
    ObjectPredicateGrid,
    ObjectSceneEvidence,
    ObjectSupportGapKind,
    ObjectSupportVersionSpace,
    build_object_support_version_space,
    cold_verify_object_support_version_space,
    evaluate_object_profile_candidate,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUNNER_ID = "bongard.object-task/pure-python-freeze-query-v1"
FREEZE_SCHEMA = "gkm.bongard-object-task-freeze.v1"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-object-task-freeze-commit.v1"
QUERY_RESULT_SCHEMA = "gkm.bongard-object-task-query-result.v1"
ARCHIVE_SCHEMA = "gkm.bongard-object-task-run-archive.v1"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardTaskRunnerError(RuntimeError):
    """A support, freeze, query-release, or replay boundary failed closed."""


class ObjectBongardTaskRunStatus(str, Enum):
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
        "lean_affects_identity_selection_decision_or_replay": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
    }


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardTaskRunnerError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardTaskRunnerError(f"{label} must be a sha256: address")
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardTaskRunnerError(f"{label} must be a bounded identifier")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectBongardTaskRunnerError(f"{label} fields differ from schema")
    return value


def object_bongard_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _support_digest(
    side_0: Sequence[ObjectSceneEvidence], side_1: Sequence[ObjectSceneEvidence]
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-task-support.v1",
            "side_0_scene_evidence": [item.to_data() for item in side_0],
            "side_1_scene_evidence": [item.to_data() for item in side_1],
            "labels_supplied_to_python_only": True,
            "query_material_included": False,
            **_authority_data(),
        }
    )


def _canonical_support(
    plan: ObjectBongardTaskPlan,
    side_0: Sequence[ObjectSceneEvidence],
    side_1: Sequence[ObjectSceneEvidence],
) -> tuple[tuple[ObjectSceneEvidence, ...], tuple[ObjectSceneEvidence, ...]]:
    positives = tuple(
        sorted(
            (ObjectSceneEvidence.from_data(item.to_data()) for item in side_0),
            key=lambda item: item.scene_id,
        )
    )
    negatives = tuple(
        sorted(
            (ObjectSceneEvidence.from_data(item.to_data()) for item in side_1),
            key=lambda item: item.scene_id,
        )
    )
    if (
        len(positives) != 6
        or len(negatives) != 6
        or tuple(item.scene_id for item in positives)
        != plan.side_0_support_panel_ids
        or tuple(item.scene_id for item in negatives)
        != plan.side_1_support_panel_ids
    ):
        raise ObjectBongardTaskRunnerError(
            "support evidence must match the exact sealed 6+6 panel identities"
        )
    catalogs = {item.lineage_catalog_digest for item in (*positives, *negatives)}
    if len(catalogs) != 1:
        raise ObjectBongardTaskRunnerError(
            "support evidence must share one lineage catalog identity"
        )
    return positives, negatives


def _canonical_parents(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    execution_precommit_digest: str,
) -> tuple[ObjectBongardTaskPlan, ObjectBongardSemanticArtifact, str]:
    plan = ObjectBongardTaskPlan.from_data(task_plan.to_data())
    semantic = ObjectBongardSemanticArtifact.from_data(semantic_artifact.to_data())
    precommit = _address(execution_precommit_digest, "execution precommit digest")
    if (
        semantic.status is not PrototypeSceneObserverStatus.SUCCESS
        or semantic.task_id != plan.task_id
        or semantic.group_panel_ids
        != (plan.side_0_support_panel_ids, plan.side_1_support_panel_ids)
        or semantic.observation_context_digest != precommit
    ):
        raise ObjectBongardTaskRunnerError(
            "successful semantic artifact does not bind the task support precommit"
        )
    return plan, semantic, precommit


def _freeze_content(value: "ObjectBongardTaskFreeze") -> dict[str, object]:
    return {
        "schema": FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "support_digest": value.support_digest,
        "lineage_catalog_digest": value.lineage_catalog_digest,
        "grid_digest": value.grid_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_input_digest": value.rank_input_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_profile": value.selected_profile.to_data(),
        "selected_profile_digest": value.selected_profile.profile_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_pixels_included": False,
        "query_evidence_included": False,
        "formula_frozen_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardTaskFreeze:
    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    semantic_artifact_digest: str
    support_digest: str
    lineage_catalog_digest: str
    grid_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_input_digest: str
    rank_response_digest: str
    selected_profile: ObjectProfile
    selected_predicate_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.runner_source_digest, "runner source digest")
        _identifier(self.task_id, "task ID")
        _address(self.task_plan_digest, "task plan digest")
        _address(self.execution_precommit_digest, "execution precommit digest")
        for name in (
            "semantic_artifact_digest",
            "support_digest",
            "lineage_catalog_digest",
            "grid_digest",
            "version_space_digest",
            "support_version_space_digest",
            "rank_input_digest",
            "rank_response_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.record_digest, "freeze record digest")
        if (
            self.runner_source_digest != object_bongard_task_runner_source_digest()
            or not isinstance(self.selected_profile, ObjectProfile)
            or self.selected_predicate_digest != self.selected_profile.profile_digest
            or self.support_version_space_digest != self.version_space_digest
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(not isinstance(item, str) or not item for item in self.sealed_query_panel_ids)
            or any(
                atom.operator is not ObjectProfileOperator.AT_LEAST
                for atom in self.selected_profile.atoms
            )
            or self.record_digest != _content_address(_freeze_content(self))
        ):
            raise ObjectBongardTaskRunnerError("task freeze differs")

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        semantic_artifact: ObjectBongardSemanticArtifact,
        support_digest: str,
        lineage_catalog_digest: str,
        grid: ObjectPredicateGrid,
        version_space: ObjectSupportVersionSpace,
        rank_input_digest: str,
        rank_response: ObjectBongardRankResponse,
        selected_profile: ObjectProfile,
    ) -> "ObjectBongardTaskFreeze":
        values: dict[str, object] = {
            "runner_source_digest": object_bongard_task_runner_source_digest(),
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.record_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "semantic_artifact_digest": semantic_artifact.artifact_digest,
            "support_digest": support_digest,
            "lineage_catalog_digest": lineage_catalog_digest,
            "grid_digest": grid.grid_digest,
            "version_space_digest": version_space.version_space_digest,
            "support_version_space_digest": version_space.version_space_digest,
            "rank_input_digest": rank_input_digest,
            "rank_response_digest": rank_response.response_digest,
            "selected_profile": selected_profile,
            "selected_predicate_digest": selected_profile.profile_digest,
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
    def from_data(cls, value: object) -> "ObjectBongardTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact_digest", "support_digest",
                "lineage_catalog_digest", "grid_digest", "version_space_digest",
                "support_version_space_digest", "rank_input_digest",
                "rank_response_digest", "selected_profile",
                "selected_profile_digest", "selected_predicate_digest",
                "sealed_query_panel_ids", "query_pixels_included",
                "query_evidence_included", "formula_frozen_before_query_release",
                *_authority_data(), "record_digest",
            },
            "object task freeze",
        )
        if (
            raw["schema"] != FREEZE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["query_pixels_included"] is not False
            or raw["query_evidence_included"] is not False
            or raw["formula_frozen_before_query_release"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["sealed_query_panel_ids"], list)
        ):
            raise ObjectBongardTaskRunnerError("task freeze policy differs")
        profile = ObjectProfile.from_data(raw["selected_profile"])
        if raw["selected_profile_digest"] != profile.profile_digest:
            raise ObjectBongardTaskRunnerError("selected profile digest differs")
        result = cls(
            raw["runner_source_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["semantic_artifact_digest"],
            raw["support_digest"], raw["lineage_catalog_digest"], raw["grid_digest"],
            raw["version_space_digest"], raw["support_version_space_digest"],
            raw["rank_input_digest"], raw["rank_response_digest"], profile,
            raw["selected_predicate_digest"], tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardTaskRunnerError("task freeze is not canonical")
        return result


def _commit_content(value: "ObjectBongardTaskFreezeCommit") -> dict[str, object]:
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
        "durable_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardTaskFreezeCommit:
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
        _address(self.execution_precommit_digest, "commit execution precommit digest")
        for name in (
            "version_space_digest", "support_version_space_digest",
            "rank_response_digest", "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        for name in (
            "task_freeze_digest", "exact_freeze_payload_digest",
            "task_freeze_store_receipt_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        if (
            self.version_space_digest != self.support_version_space_digest
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise ObjectBongardTaskRunnerError("freeze commit differs")

    @classmethod
    def seal(
        cls,
        freeze: ObjectBongardTaskFreeze,
        exact_freeze_payload: bytes,
        *,
        task_freeze_store_receipt_digest: str,
    ) -> "ObjectBongardTaskFreezeCommit":
        expected = canonical_json(freeze.to_data()) + b"\n"
        if exact_freeze_payload != expected:
            raise ObjectBongardTaskRunnerError("freeze payload bytes are not canonical")
        values: dict[str, object] = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "rank_response_digest": freeze.rank_response_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": "sha256:" + hashlib.sha256(expected).hexdigest(),
            "task_freeze_store_receipt_digest": task_freeze_store_receipt_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_commit_content(provisional)),
        )

    def assert_matches(
        self, freeze: ObjectBongardTaskFreeze, exact_freeze_payload: bytes
    ) -> None:
        if self != type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=self.task_freeze_store_receipt_digest,
        ):
            raise ObjectBongardTaskRunnerError("freeze commit replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "task_id", "task_plan_digest",
                "execution_precommit_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "task_freeze_store_receipt_digest",
                "durable_before_query_release",
                *_authority_data(), "record_digest",
            },
            "object task freeze commit",
        )
        if (
            raw["schema"] != FREEZE_COMMIT_SCHEMA
            or raw["durable_before_query_release"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardTaskRunnerError("freeze commit policy differs")
        result = cls(
            raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["rank_response_digest"],
            raw["selected_predicate_digest"], raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"],
            raw["task_freeze_store_receipt_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardTaskRunnerError("freeze commit is not canonical")
        return result


def _query_result_content(value: "ObjectBongardTaskQueryResult") -> dict[str, object]:
    return {
        "schema": QUERY_RESULT_SCHEMA,
        "side": value.side,
        "scene_id": value.scene_id,
        "expected_disposition": value.expected_disposition.value,
        "evaluation": value.evaluation.to_data(),
        "correct": value.correct,
        "abstained": value.abstained,
        "fixed_denominator_contribution": 1,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardTaskQueryResult:
    side: str
    scene_id: str
    expected_disposition: Disposition
    evaluation: ObjectCandidateSceneEvaluation
    correct: bool
    abstained: bool
    result_digest: str

    def __post_init__(self) -> None:
        if self.side not in ("side_0", "side_1"):
            raise ObjectBongardTaskRunnerError("query side is unknown")
        expected = (
            Disposition.PRESENT
            if self.side == "side_0"
            else Disposition.CERTIFIED_ABSENT
        )
        if (
            self.expected_disposition is not expected
            or self.scene_id == ""
            or self.correct is not (self.evaluation.disposition is expected)
            or self.abstained
            is not (
                self.evaluation.disposition
                in (Disposition.INDETERMINATE, Disposition.ERROR)
            )
            or self.result_digest != canonical_digest(_query_result_content(self))
        ):
            raise ObjectBongardTaskRunnerError("query result differs from fixed scoring")

    @classmethod
    def create(
        cls,
        side: str,
        scene: ObjectSceneEvidence,
        evaluation: ObjectCandidateSceneEvaluation,
    ) -> "ObjectBongardTaskQueryResult":
        expected = (
            Disposition.PRESENT if side == "side_0" else Disposition.CERTIFIED_ABSENT
        )
        values: dict[str, object] = {
            "side": side,
            "scene_id": scene.scene_id,
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
    def from_data(cls, value: object) -> "ObjectBongardTaskQueryResult":
        raw = _fields(
            value,
            {
                "schema", "side", "scene_id", "expected_disposition",
                "evaluation", "correct", "abstained",
                "fixed_denominator_contribution", *_authority_data(), "result_digest",
            },
            "object task query result",
        )
        if (
            raw["schema"] != QUERY_RESULT_SCHEMA
            or raw["fixed_denominator_contribution"] != 1
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardTaskRunnerError("query result policy differs")
        try:
            expected = Disposition(raw["expected_disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardTaskRunnerError("query expected disposition differs") from exc
        result = cls(
            raw["side"], raw["scene_id"], expected,
            ObjectCandidateSceneEvaluation.from_data(raw["evaluation"]),
            raw["correct"], raw["abstained"], raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardTaskRunnerError("query result is not canonical")
        return result


def _archive_content(value: "ObjectBongardTaskRunArchive") -> dict[str, object]:
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
        "side_0_support": [item.to_data() for item in value.side_0_support],
        "side_1_support": [item.to_data() for item in value.side_1_support],
        "support_digest": value.support_digest,
        "grid": value.grid.to_data(),
        "version_space": value.version_space.to_data(),
        "version_space_digest": value.version_space.version_space_digest,
        "rank_input_digest": value.rank_input_digest,
        "rank_response": None if value.rank_response is None else value.rank_response.to_data(),
        "freeze": None if value.freeze is None else value.freeze.to_data(),
        "freeze_commit": None if value.freeze_commit is None else value.freeze_commit.to_data(),
        "side_0_query": None if value.side_0_query is None else value.side_0_query.to_data(),
        "side_1_query": None if value.side_1_query is None else value.side_1_query.to_data(),
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
class ObjectBongardTaskRunArchive:
    status: ObjectBongardTaskRunStatus
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    semantic_artifact: ObjectBongardSemanticArtifact
    side_0_support: tuple[ObjectSceneEvidence, ...]
    side_1_support: tuple[ObjectSceneEvidence, ...]
    support_digest: str
    grid: ObjectPredicateGrid
    version_space: ObjectSupportVersionSpace
    rank_input_digest: str | None
    rank_response: ObjectBongardRankResponse | None
    freeze: ObjectBongardTaskFreeze | None
    freeze_commit: ObjectBongardTaskFreezeCommit | None
    side_0_query: ObjectSceneEvidence | None
    side_1_query: ObjectSceneEvidence | None
    query_results: tuple[ObjectBongardTaskQueryResult, ...]
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
        if not isinstance(self.status, ObjectBongardTaskRunStatus):
            raise TypeError("task run status has the wrong type")
        if self.runner_source_digest != object_bongard_task_runner_source_digest():
            raise ObjectBongardTaskRunnerError("archive runner source differs")
        plan, semantic, precommit = _canonical_parents(
            self.task_plan, self.semantic_artifact, self.execution_precommit_digest
        )
        positives, negatives = _canonical_support(
            plan, self.side_0_support, self.side_1_support
        )
        if plan != self.task_plan or semantic != self.semantic_artifact or precommit != self.execution_precommit_digest:
            raise ObjectBongardTaskRunnerError("archive parents are not canonical")
        if self.support_digest != _support_digest(positives, negatives):
            raise ObjectBongardTaskRunnerError("archive support digest differs")
        expected_grid = ObjectPredicateGrid.create(semantic.feature_families[0])
        replayed = cold_verify_object_support_version_space(
            self.version_space, expected_grid, positives, negatives
        )
        if self.grid != expected_grid or replayed != self.version_space:
            raise ObjectBongardTaskRunnerError("archive version-space replay differs")
        if not self.version_space.survivor_profile_digests:
            assert self.version_space.gap is not None
            expected_status = (
                ObjectBongardTaskRunStatus.LANGUAGE_GAP
                if self.version_space.gap.kind is ObjectSupportGapKind.LANGUAGE_GAP
                else ObjectBongardTaskRunStatus.WITNESS_GAP
            )
            if (
                self.status is not expected_status
                or any(item is not None for item in (
                    self.rank_input_digest, self.rank_response, self.freeze,
                    self.freeze_commit, self.side_0_query, self.side_1_query,
                    self.accuracy_ppm,
                ))
                or self.query_results
                or any((self.correct_count, self.abstention_count, self.score_denominator,
                        self.rank_calls_made, self.rank_verification_calls_made,
                        self.freeze_commit_calls_made, self.freeze_reload_calls_made,
                        self.query_source_calls_made))
            ):
                raise ObjectBongardTaskRunnerError("gap archive crossed a later phase")
        else:
            if (
                self.status is not ObjectBongardTaskRunStatus.COMPLETE
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
                or self.freeze_reload_calls_made not in (0, 1)
                or self.query_source_calls_made != 1
            ):
                raise ObjectBongardTaskRunnerError("complete archive phase counts differ")
            survivors = tuple(
                self.version_space.survivor(item)
                for item in self.version_space.survivor_profile_digests
            )
            expected_rank_input = object_bongard_rank_input_digest(
                survivors=survivors,
                neutral_rubrics=semantic.rubrics,
                feature_nominations=semantic.feature_families,
                semantic_artifact_digest=semantic.artifact_digest,
                version_space_digest=self.version_space.version_space_digest,
            )
            self.rank_response.assert_matches(
                survivor_profile_digests=self.version_space.survivor_profile_digests,
                rank_input_digest=expected_rank_input,
            )
            selected = self.version_space.survivor(
                self.rank_response.selected_profile_digest
            )
            expected_freeze = ObjectBongardTaskFreeze.seal(
                task_plan=plan,
                execution_precommit_digest=precommit,
                semantic_artifact=semantic,
                support_digest=self.support_digest,
                lineage_catalog_digest=positives[0].lineage_catalog_digest,
                grid=self.grid,
                version_space=self.version_space,
                rank_input_digest=expected_rank_input,
                rank_response=self.rank_response,
                selected_profile=selected,
            )
            freeze_bytes = canonical_json(expected_freeze.to_data()) + b"\n"
            self.freeze_commit.assert_matches(expected_freeze, freeze_bytes)
            queries = (self.side_0_query, self.side_1_query)
            if (
                self.rank_input_digest != expected_rank_input
                or self.freeze != expected_freeze
                or tuple(item.scene_id for item in queries)
                != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
                or any(item.lineage_catalog_digest != positives[0].lineage_catalog_digest for item in queries)
            ):
                raise ObjectBongardTaskRunnerError("archive rank, freeze, or query binding differs")
            evaluations = tuple(
                evaluate_object_profile_candidate(self.grid, selected, scene)
                for scene in queries
            )
            expected_results = tuple(
                ObjectBongardTaskQueryResult.create(side, scene, evaluation)
                for side, scene, evaluation in zip(
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
                raise ObjectBongardTaskRunnerError("archive fixed query score differs")
        _raw_digest(self.record_digest, "archive digest")
        if self.record_digest != canonical_digest(_archive_content(self)):
            raise ObjectBongardTaskRunnerError("archive digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "status", "runner_source_digest",
                "task_plan", "task_plan_digest", "execution_precommit_digest",
                "semantic_artifact", "semantic_artifact_digest", "side_0_support",
                "side_1_support", "support_digest", "grid", "version_space",
                "version_space_digest", "rank_input_digest", "rank_response",
                "freeze", "freeze_commit", "side_0_query", "side_1_query",
                "query_results", "correct_count", "abstention_count",
                "score_denominator", "accuracy_ppm", "rank_calls_made",
                "rank_verification_calls_made", "freeze_commit_calls_made",
                "freeze_reload_calls_made", "query_source_calls_made",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "object task run archive",
        )
        if (
            raw["schema"] != ARCHIVE_SCHEMA
            or raw["runner_id"] != RUNNER_ID
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(not isinstance(raw[name], list) for name in (
                "side_0_support", "side_1_support", "query_results"
            ))
        ):
            raise ObjectBongardTaskRunnerError("archive policy differs")
        plan = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        semantic = ObjectBongardSemanticArtifact.from_data(raw["semantic_artifact"])
        version = ObjectSupportVersionSpace.from_data(raw["version_space"])
        if (
            raw["task_plan_digest"] != plan.record_digest
            or raw["semantic_artifact_digest"] != semantic.artifact_digest
            or raw["version_space_digest"] != version.version_space_digest
        ):
            raise ObjectBongardTaskRunnerError("archive parent digest differs")
        result = cls(
            ObjectBongardTaskRunStatus(raw["status"]), raw["runner_source_digest"],
            plan, raw["execution_precommit_digest"], semantic,
            tuple(ObjectSceneEvidence.from_data(item) for item in raw["side_0_support"]),
            tuple(ObjectSceneEvidence.from_data(item) for item in raw["side_1_support"]),
            raw["support_digest"], ObjectPredicateGrid.from_data(raw["grid"]), version,
            raw["rank_input_digest"],
            None if raw["rank_response"] is None else ObjectBongardRankResponse.from_data(raw["rank_response"]),
            None if raw["freeze"] is None else ObjectBongardTaskFreeze.from_data(raw["freeze"]),
            None if raw["freeze_commit"] is None else ObjectBongardTaskFreezeCommit.from_data(raw["freeze_commit"]),
            None if raw["side_0_query"] is None else ObjectSceneEvidence.from_data(raw["side_0_query"]),
            None if raw["side_1_query"] is None else ObjectSceneEvidence.from_data(raw["side_1_query"]),
            tuple(ObjectBongardTaskQueryResult.from_data(item) for item in raw["query_results"]),
            raw["correct_count"], raw["abstention_count"], raw["score_denominator"],
            raw["accuracy_ppm"], raw["rank_calls_made"],
            raw["rank_verification_calls_made"], raw["freeze_commit_calls_made"],
            raw["freeze_reload_calls_made"], raw["query_source_calls_made"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardTaskRunnerError("archive is not canonical")
        return result


class ObjectBongardTaskRanker(Protocol):
    def __call__(self, survivors: Sequence[ObjectProfile], **kwargs: object) -> object: ...
    def verify_response(self, response: ObjectBongardRankResponse, **kwargs: object) -> object: ...


FreezeCommitter = Callable[[bytes], ObjectBongardTaskFreezeCommit | Mapping[str, Any]]
FreezeReloader = Callable[[Mapping[str, object]], bytes]
QuerySource = Callable[
    [Mapping[str, object], Mapping[str, object]], Mapping[str, ObjectSceneEvidence]
]


def _make_archive(**values: object) -> ObjectBongardTaskRunArchive:
    provisional = object.__new__(ObjectBongardTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardTaskRunArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_content(provisional)),
    )


def run_object_bongard_task(
    task_plan: ObjectBongardTaskPlan,
    semantic_artifact: ObjectBongardSemanticArtifact,
    side_0_support: Sequence[ObjectSceneEvidence],
    side_1_support: Sequence[ObjectSceneEvidence],
    *,
    execution_precommit_digest: str,
    ranker: ObjectBongardTaskRanker,
    freeze_committer: FreezeCommitter,
    query_source: QuerySource,
    freeze_reloader: FreezeReloader | None = None,
) -> ObjectBongardTaskRunArchive:
    """Run support synthesis, freeze, and exactly one 1+1 query release."""

    plan, semantic, precommit = _canonical_parents(
        task_plan, semantic_artifact, execution_precommit_digest
    )
    positives, negatives = _canonical_support(plan, side_0_support, side_1_support)
    grid = ObjectPredicateGrid.create(semantic.feature_families[0])
    version = build_object_support_version_space(grid, positives, negatives)
    version = cold_verify_object_support_version_space(
        version, grid, positives, negatives
    )
    support_digest = _support_digest(positives, negatives)
    common: dict[str, object] = {
        "runner_source_digest": object_bongard_task_runner_source_digest(),
        "task_plan": plan,
        "execution_precommit_digest": precommit,
        "semantic_artifact": semantic,
        "side_0_support": positives,
        "side_1_support": negatives,
        "support_digest": support_digest,
        "grid": grid,
        "version_space": version,
    }
    if not version.survivor_profile_digests:
        assert version.gap is not None
        return _make_archive(
            status=(
                ObjectBongardTaskRunStatus.LANGUAGE_GAP
                if version.gap.kind is ObjectSupportGapKind.LANGUAGE_GAP
                else ObjectBongardTaskRunStatus.WITNESS_GAP
            ),
            **common,
            rank_input_digest=None, rank_response=None, freeze=None,
            freeze_commit=None, side_0_query=None, side_1_query=None,
            query_results=(), correct_count=0, abstention_count=0,
            score_denominator=0, accuracy_ppm=None, rank_calls_made=0,
            rank_verification_calls_made=0, freeze_commit_calls_made=0,
            freeze_reload_calls_made=0, query_source_calls_made=0,
        )
    survivors = tuple(
        version.survivor(item) for item in version.survivor_profile_digests
    )
    rank_input = object_bongard_rank_input_digest(
        survivors=survivors,
        neutral_rubrics=semantic.rubrics,
        feature_nominations=semantic.feature_families,
        semantic_artifact_digest=semantic.artifact_digest,
        version_space_digest=version.version_space_digest,
    )
    raw_response = ranker(
        survivors,
        neutral_rubrics=semantic.rubrics,
        feature_nominations=semantic.feature_families,
        semantic_artifact_digest=semantic.artifact_digest,
        version_space_digest=version.version_space_digest,
        rank_input_digest=rank_input,
    )
    response = (
        raw_response
        if isinstance(raw_response, ObjectBongardRankResponse)
        else ObjectBongardRankResponse.from_data(raw_response)
    )
    ranker.verify_response(
        response,
        survivors=survivors,
        neutral_rubrics=semantic.rubrics,
        feature_nominations=semantic.feature_families,
        semantic_artifact_digest=semantic.artifact_digest,
        version_space_digest=version.version_space_digest,
        rank_input_digest=rank_input,
        expected_response_digest=response.response_digest,
    )
    response.assert_matches(
        survivor_profile_digests=version.survivor_profile_digests,
        rank_input_digest=rank_input,
    )
    selected = version.survivor(response.selected_profile_digest)
    freeze = ObjectBongardTaskFreeze.seal(
        task_plan=plan,
        execution_precommit_digest=precommit,
        semantic_artifact=semantic,
        support_digest=support_digest,
        lineage_catalog_digest=positives[0].lineage_catalog_digest,
        grid=grid,
        version_space=version,
        rank_input_digest=rank_input,
        rank_response=response,
        selected_profile=selected,
    )
    freeze_data = ObjectBongardTaskFreeze.from_data(freeze.to_data()).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, ObjectBongardTaskFreezeCommit)
        else ObjectBongardTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)
    reload_calls = 0
    if freeze_reloader is not None:
        reloaded = freeze_reloader(commit.to_data())
        reload_calls = 1
        if reloaded != freeze_bytes:
            raise ObjectBongardTaskRunnerError("durable freeze reload differs")
        ObjectBongardTaskFreeze.from_data(
            json.loads(reloaded.decode("utf-8"))
        )
    raw_queries = query_source(freeze_data, commit.to_data())
    if not isinstance(raw_queries, Mapping) or set(raw_queries) != {"side_0", "side_1"}:
        raise ObjectBongardTaskRunnerError("query source must return exact side_0+side_1")
    queries = tuple(
        ObjectSceneEvidence.from_data(raw_queries[side].to_data())
        for side in ("side_0", "side_1")
    )
    if (
        tuple(item.scene_id for item in queries)
        != (plan.side_0_query_panel_id, plan.side_1_query_panel_id)
        or any(item.lineage_catalog_digest != positives[0].lineage_catalog_digest for item in queries)
    ):
        raise ObjectBongardTaskRunnerError("query evidence differs from the sealed identities")
    evaluations = tuple(
        evaluate_object_profile_candidate(grid, selected, scene) for scene in queries
    )
    results = tuple(
        ObjectBongardTaskQueryResult.create(side, scene, evaluation)
        for side, scene, evaluation in zip(
            ("side_0", "side_1"), queries, evaluations, strict=True
        )
    )
    correct = sum(item.correct for item in results)
    return _make_archive(
        status=ObjectBongardTaskRunStatus.COMPLETE,
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
        freeze_reload_calls_made=reload_calls,
        query_source_calls_made=1,
    )


def cold_replay_object_bongard_task(
    archive: ObjectBongardTaskRunArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> ObjectBongardTaskRunArchive:
    """Replay the complete archive with no rank, persistence, query, or model call."""

    expected = _raw_digest(expected_archive_digest, "expected archive digest")
    supplied = (
        archive.record_digest
        if isinstance(archive, ObjectBongardTaskRunArchive)
        else archive.get("record_digest")
    )
    if supplied != expected:
        raise ObjectBongardTaskRunnerError("archive differs from external commitment")
    restored = (
        ObjectBongardTaskRunArchive.from_data(archive.to_data())
        if isinstance(archive, ObjectBongardTaskRunArchive)
        else ObjectBongardTaskRunArchive.from_data(archive)
    )
    if restored.record_digest != expected:
        raise ObjectBongardTaskRunnerError("cold archive digest differs")
    return restored


__all__ = (
    "ARCHIVE_SCHEMA", "FREEZE_COMMIT_SCHEMA", "FREEZE_SCHEMA", "QUERY_RESULT_SCHEMA",
    "RUNNER_ID", "ObjectBongardTaskFreeze", "ObjectBongardTaskFreezeCommit",
    "ObjectBongardTaskQueryResult", "ObjectBongardTaskRunArchive",
    "ObjectBongardTaskRunStatus", "ObjectBongardTaskRunnerError",
    "cold_replay_object_bongard_task", "object_bongard_task_runner_source_digest",
    "run_object_bongard_task",
)
