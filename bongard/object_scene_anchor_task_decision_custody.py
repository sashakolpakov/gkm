"""Pure-Python task-decision custody for the anchor predicate pipeline.

This module is deliberately below every runner.  It performs no model call,
query access, persistence, or filesystem operation.  It validates exact parent
artifacts, freezes the selected Python predicate before query release, and
binds the canonical persisted freeze payload to the release-store receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
    ObjectBongardWriteOnceReceipt,
)
from bongard.object_scene_anchor_batch_observer import (
    object_scene_anchor_batch_observer_prompt,
)
from bongard.object_scene_anchor_candidate_ranker import (
    ObjectSceneAnchorRankInput,
    ObjectSceneAnchorRankResponse,
    freeze_object_scene_anchor_rank_input,
    object_scene_anchor_candidate_ranker_prompt,
)
from bongard.object_scene_anchor_python_bridge import (
    ObjectSceneAnchorPythonBridgeArtifact,
    cold_verify_object_scene_anchor_python_bridge,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
)
from bongard.object_scene_anchor_support_observation_join import (
    ObjectSceneAnchorSupportObservationPlan,
    ObjectSceneAnchorSupportObservationResult,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorSupportVersionSpace,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)


OBJECT_SCENE_ANCHOR_TASK_DECISION_FREEZE_SCHEMA = (
    "gkm.object-scene-anchor-task-decision-freeze.v1"
)
OBJECT_SCENE_ANCHOR_TASK_DECISION_COMMIT_SCHEMA = (
    "gkm.object-scene-anchor-task-decision-commit.v1"
)
OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID = (
    "bongard.object-scene-anchor-task-decision-custody/python-v1"
)

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ObjectSceneAnchorTaskDecisionCustodyError(ValueError):
    """A task freeze, durable commit, parent, or replay differs."""


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            f"{label} must be a sha256: address"
        )
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(f"{label} fields differ")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "pure_python_predicate_execution": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_release_or_replay": False,
        "legacy_scene_runner_reused": False,
        "rubric_runner_reused": False,
        "model_calls_permitted": False,
        "filesystem_io_permitted": False,
    }


def object_scene_anchor_task_decision_custody_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-task-decision-custody-algorithm.v1",
            "algorithm_id": OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID,
            "source_digest": (
                object_scene_anchor_task_decision_custody_source_digest()
            ),
            "support_rule": "exact-task-ordered-six-plus-six",
            "rank_rule": "exact-union-of-every-nonempty-orientation",
            "empty_orientation_rule": "exact-typed-gap-omission-proof",
            "selection_rule": "exact-rank-response-to-positive-python-predicate",
            "release_digest_rule": "rank-scope-and-response-and-predicate-raw",
            "query_rule": "identities-and-pixels-absent-from-model-artifacts",
            **_authority_data(),
        }
    )


def object_scene_anchor_task_decision_custody_source_digest() -> str:
    """Return the import-time custody source digest after drift verification."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _has_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, Mapping):
        return any(_has_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_bytes(item) for item in value)
    return False


def _contains_text(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(
            _contains_text(key, needle) or _contains_text(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_text(item, needle) for item in value)
    return False


def _opposite(orientation: ObjectSceneAnchorOrientation) -> ObjectSceneAnchorOrientation:
    return (
        ObjectSceneAnchorOrientation.SIDE1_POSITIVE
        if orientation is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
        else ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )


def _predicate_mapping(predicate: ObjectSceneAnchorPythonPredicate) -> dict[str, object]:
    orientation = predicate.selection_commitment.orientation
    return {
        "schema": "gkm.object-scene-anchor-selected-python-predicate-mapping.v1",
        "predicate_digest": predicate.predicate_digest,
        "selection_commitment_digest": (
            predicate.selection_commitment.selection_commitment_digest
        ),
        "origin_version_space_digest": predicate.version_space_digest,
        "selected_candidate_digest": predicate.candidate.candidate_digest,
        "orientation": orientation.value,
        "formula_digest": predicate.formula.formula_digest,
        "witness_digests": list(predicate.formula.witness_digests),
        "present_bucket": orientation.value,
        "certified_absent_bucket": _opposite(orientation).value,
        "indeterminate_bucket": "abstain",
        "error_bucket": "error",
        "positive_witnesses_only": True,
        "negation_available": False,
        "polarity_flip_available": False,
        "same_binding_required": True,
        "pure_python_evaluation": True,
    }


def _omitted_space(
    bridge: ObjectSceneAnchorPythonBridgeArtifact,
) -> ObjectSceneAnchorSupportVersionSpace | None:
    return bridge.omitted_version_space


def _freeze_content(value: "ObjectSceneAnchorTaskDecisionFreeze") -> dict[str, object]:
    omitted = _omitted_space(value.bridge)
    return {
        "schema": OBJECT_SCENE_ANCHOR_TASK_DECISION_FREEZE_SCHEMA,
        "custody_id": OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID,
        "algorithm_digest": value.algorithm_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "support_observation_plan_digest": value.support_observation_plan_digest,
        "support_observation_result_digest": value.support_observation_result_digest,
        "support_corpus_freeze_digest": value.support_corpus_freeze_digest,
        "batch_artifact_digest": value.batch_artifact_digest,
        "language_digest": value.language_digest,
        "support_panel_ids": list(value.support_panel_ids),
        "support_panel_binding_digests": list(value.support_panel_binding_digests),
        "support_panel_png_digests": list(value.support_panel_png_digests),
        "support_evaluation_digests": list(value.support_evaluation_digests),
        "orientation_version_space_digests": list(
            value.orientation_version_space_digests
        ),
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_input": value.rank_input.to_data(),
        "rank_input_digest": value.rank_input_digest,
        "ranked_child_version_space_digests": list(
            value.rank_input.child_version_space_digests
        ),
        "ranked_child_orientations": list(value.rank_input.child_orientations),
        "rank_response": value.rank_response.to_data(),
        "rank_response_digest": value.rank_response_digest,
        "rank_response_output_digest": value.rank_response.output_digest,
        "rank_response_receipt_digest": value.rank_response.receipt_digest,
        "bridge": value.bridge.to_data(),
        "bridge_digest": value.bridge_digest,
        "omitted_gap_version_space": (
            None if omitted is None else omitted.to_data()
        ),
        "omitted_gap_version_space_digest": (
            None if omitted is None else omitted.version_space_digest
        ),
        "omitted_gap_digest": (
            None if omitted is None else omitted.gap.gap_digest
        ),
        "selected_predicate": value.selected_predicate.to_data(),
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_python_predicate_mapping": _predicate_mapping(
            value.selected_predicate
        ),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_bytes_included": False,
        "query_labels_included": False,
        "query_identities_model_visible": False,
        "formula_frozen_before_query_release": True,
        "support_observation_result_bound": True,
        "complete_nonempty_rank_scope_bound": True,
        "rank_response_and_bridge_bound": True,
        "typed_gap_omission_proof_required": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskDecisionFreeze:
    """Exact support-to-predicate decision frozen before query release."""

    algorithm_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    support_observation_plan_digest: str
    support_observation_result_digest: str
    support_corpus_freeze_digest: str
    batch_artifact_digest: str
    language_digest: str
    support_panel_ids: tuple[str, ...]
    support_panel_binding_digests: tuple[str, ...]
    support_panel_png_digests: tuple[str, ...]
    support_evaluation_digests: tuple[str, ...]
    orientation_version_space_digests: tuple[str, str]
    version_space_digest: str
    support_version_space_digest: str
    rank_input: ObjectSceneAnchorRankInput
    rank_input_digest: str
    rank_response: ObjectSceneAnchorRankResponse
    rank_response_digest: str
    bridge: ObjectSceneAnchorPythonBridgeArtifact
    bridge_digest: str
    selected_predicate: ObjectSceneAnchorPythonPredicate
    selected_predicate_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("algorithm digest", self.algorithm_digest),
            ("support observation plan digest", self.support_observation_plan_digest),
            ("support observation result digest", self.support_observation_result_digest),
            ("support corpus freeze digest", self.support_corpus_freeze_digest),
            ("batch artifact digest", self.batch_artifact_digest),
            ("language digest", self.language_digest),
            ("version-space digest", self.version_space_digest),
            ("support version-space digest", self.support_version_space_digest),
            ("rank input digest", self.rank_input_digest),
            ("rank response digest", self.rank_response_digest),
            ("bridge digest", self.bridge_digest),
            ("selected predicate digest", self.selected_predicate_digest),
        ):
            _raw_digest(item, label)
        _address(self.task_plan_digest, "task plan digest")
        _address(self.execution_precommit_digest, "execution precommit digest")
        _address(self.record_digest, "task decision freeze digest")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ObjectSceneAnchorTaskDecisionCustodyError("task ID differs")
        for values, label, count in (
            (self.support_panel_ids, "support panel IDs", 12),
            (self.support_panel_binding_digests, "support panel bindings", 12),
            (self.support_panel_png_digests, "support panel PNG digests", 12),
            (self.support_evaluation_digests, "support evaluations", 12),
        ):
            if type(values) is not tuple or len(values) != count:
                raise ObjectSceneAnchorTaskDecisionCustodyError(f"{label} differ")
        if len(set(self.support_panel_ids)) != 12:
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "support panel IDs are not unique"
            )
        for item in (
            *self.support_panel_binding_digests,
            *self.support_panel_png_digests,
            *self.support_evaluation_digests,
            *self.orientation_version_space_digests,
        ):
            _raw_digest(item, "task decision child digest")
        if (
            type(self.orientation_version_space_digests) is not tuple
            or len(self.orientation_version_space_digests) != 2
            or len(set(self.orientation_version_space_digests)) != 2
            or type(self.sealed_query_panel_ids) is not tuple
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or set(self.support_panel_ids) & set(self.sealed_query_panel_ids)
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision panel or orientation inventory differs"
            )
        if type(self.rank_input) is not ObjectSceneAnchorRankInput:
            raise TypeError("rank input has the wrong type")
        if type(self.rank_response) is not ObjectSceneAnchorRankResponse:
            raise TypeError("rank response has the wrong type")
        if type(self.bridge) is not ObjectSceneAnchorPythonBridgeArtifact:
            raise TypeError("Python bridge has the wrong type")
        if type(self.selected_predicate) is not ObjectSceneAnchorPythonPredicate:
            raise TypeError("selected Python predicate has the wrong type")
        rank_input = ObjectSceneAnchorRankInput.from_data(self.rank_input.to_data())
        response = ObjectSceneAnchorRankResponse.from_data(
            self.rank_response.to_data()
        )
        bridge = ObjectSceneAnchorPythonBridgeArtifact.from_data(
            self.bridge.to_data()
        )
        predicate = ObjectSceneAnchorPythonPredicate.from_data(
            self.selected_predicate.to_data()
        )
        omitted = bridge.omitted_version_space
        covered = set(rank_input.child_version_space_digests)
        if omitted is not None:
            covered.add(omitted.version_space_digest)
        if (
            self.algorithm_digest
            != object_scene_anchor_task_decision_custody_algorithm_digest()
            or self.version_space_digest != self.support_version_space_digest
            or self.version_space_digest != rank_input.version_space_digest
            or self.rank_input_digest != rank_input.rank_input_digest
            or response.rank_input != rank_input
            or response.rank_input_digest != self.rank_input_digest
            or self.rank_response_digest != response.response_digest
            or bridge.rank_input_digest != self.rank_input_digest
            or bridge.rank_response_digest != self.rank_response_digest
            or bridge.child_version_space_digests
            != rank_input.child_version_space_digests
            or tuple(item.value for item in bridge.child_orientations)
            != rank_input.child_orientations
            or self.bridge_digest != bridge.bridge_digest
            or bridge.predicate != predicate
            or self.selected_predicate_digest != predicate.predicate_digest
            or predicate.language_digest != self.language_digest
            or predicate.version_space_digest
            not in rank_input.child_version_space_digests
            or covered != set(self.orientation_version_space_digests)
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision rank, bridge, or predicate binding differs"
            )
        if omitted is not None and (
            omitted.survivor_candidate_digests or omitted.gap is None
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "omitted orientation lacks an exact typed gap"
            )
        unsigned = _freeze_content(self)
        if _has_bytes(unsigned):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision freeze contains bytes"
            )
        if self.record_digest != _content_address(unsigned):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision freeze digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorTaskDecisionFreeze":
        raw = _fields(
            value,
            {
                "schema", "custody_id", "algorithm_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "support_observation_plan_digest",
                "support_observation_result_digest", "support_corpus_freeze_digest",
                "batch_artifact_digest", "language_digest", "support_panel_ids",
                "support_panel_binding_digests", "support_panel_png_digests",
                "support_evaluation_digests", "orientation_version_space_digests",
                "version_space_digest", "support_version_space_digest",
                "rank_input", "rank_input_digest",
                "ranked_child_version_space_digests", "ranked_child_orientations",
                "rank_response", "rank_response_digest", "rank_response_output_digest",
                "rank_response_receipt_digest", "bridge", "bridge_digest",
                "omitted_gap_version_space", "omitted_gap_version_space_digest",
                "omitted_gap_digest", "selected_predicate",
                "selected_predicate_digest", "selected_python_predicate_mapping",
                "sealed_query_panel_ids", "query_bytes_included",
                "query_labels_included", "query_identities_model_visible",
                "formula_frozen_before_query_release",
                "support_observation_result_bound",
                "complete_nonempty_rank_scope_bound",
                "rank_response_and_bridge_bound", "typed_gap_omission_proof_required",
                *_authority_data(), "record_digest",
            },
            "anchor task decision freeze",
        )
        list_fields = (
            "support_panel_ids", "support_panel_binding_digests",
            "support_panel_png_digests", "support_evaluation_digests",
            "orientation_version_space_digests", "ranked_child_version_space_digests",
            "ranked_child_orientations", "sealed_query_panel_ids",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_TASK_DECISION_FREEZE_SCHEMA
            or raw["custody_id"] != OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID
            or any(not isinstance(raw[name], list) for name in list_fields)
            or not isinstance(raw["rank_input"], Mapping)
            or not isinstance(raw["rank_response"], Mapping)
            or not isinstance(raw["bridge"], Mapping)
            or not isinstance(raw["selected_predicate"], Mapping)
            or raw["query_bytes_included"] is not False
            or raw["query_labels_included"] is not False
            or raw["query_identities_model_visible"] is not False
            or raw["formula_frozen_before_query_release"] is not True
            or raw["support_observation_result_bound"] is not True
            or raw["complete_nonempty_rank_scope_bound"] is not True
            or raw["rank_response_and_bridge_bound"] is not True
            or raw["typed_gap_omission_proof_required"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision freeze policy differs"
            )
        rank_input = ObjectSceneAnchorRankInput.from_data(raw["rank_input"])
        response = ObjectSceneAnchorRankResponse.from_data(raw["rank_response"])
        bridge = ObjectSceneAnchorPythonBridgeArtifact.from_data(raw["bridge"])
        predicate = ObjectSceneAnchorPythonPredicate.from_data(
            raw["selected_predicate"]
        )
        omitted = bridge.omitted_version_space
        if (
            raw["ranked_child_version_space_digests"]
            != list(rank_input.child_version_space_digests)
            or raw["ranked_child_orientations"] != list(rank_input.child_orientations)
            or raw["rank_response_output_digest"] != response.output_digest
            or raw["rank_response_receipt_digest"] != response.receipt_digest
            or raw["omitted_gap_version_space"]
            != (None if omitted is None else omitted.to_data())
            or raw["omitted_gap_version_space_digest"]
            != (None if omitted is None else omitted.version_space_digest)
            or raw["omitted_gap_digest"]
            != (None if omitted is None else omitted.gap.gap_digest)
            or raw["selected_python_predicate_mapping"]
            != _predicate_mapping(predicate)
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision derived commitment differs"
            )
        result = cls(
            raw["algorithm_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"],
            raw["support_observation_plan_digest"],
            raw["support_observation_result_digest"],
            raw["support_corpus_freeze_digest"], raw["batch_artifact_digest"],
            raw["language_digest"], tuple(raw["support_panel_ids"]),
            tuple(raw["support_panel_binding_digests"]),
            tuple(raw["support_panel_png_digests"]),
            tuple(raw["support_evaluation_digests"]),
            tuple(raw["orientation_version_space_digests"]),
            raw["version_space_digest"], raw["support_version_space_digest"],
            rank_input, raw["rank_input_digest"], response,
            raw["rank_response_digest"], bridge, raw["bridge_digest"],
            predicate, raw["selected_predicate_digest"],
            tuple(raw["sealed_query_panel_ids"]), raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision freeze is not canonical"
            )
        return result


def _exact_parent_artifacts(
    task: ObjectBongardTaskPlan,
    precommit: ObjectBongardExecutionPrecommit,
    support_plan: ObjectSceneAnchorSupportObservationPlan,
    support_result: ObjectSceneAnchorSupportObservationResult,
    rank_input: ObjectSceneAnchorRankInput,
    rank_response: ObjectSceneAnchorRankResponse,
    bridge: ObjectSceneAnchorPythonBridgeArtifact,
    predicate: ObjectSceneAnchorPythonPredicate,
) -> tuple[
    ObjectBongardTaskPlan,
    ObjectBongardExecutionPrecommit,
    ObjectSceneAnchorSupportObservationPlan,
    ObjectSceneAnchorSupportObservationResult,
    ObjectSceneAnchorRankInput,
    ObjectSceneAnchorRankResponse,
    ObjectSceneAnchorPythonBridgeArtifact,
    ObjectSceneAnchorPythonPredicate,
]:
    expected_types = (
        ObjectBongardTaskPlan, ObjectBongardExecutionPrecommit,
        ObjectSceneAnchorSupportObservationPlan,
        ObjectSceneAnchorSupportObservationResult, ObjectSceneAnchorRankInput,
        ObjectSceneAnchorRankResponse, ObjectSceneAnchorPythonBridgeArtifact,
        ObjectSceneAnchorPythonPredicate,
    )
    values = (
        task, precommit, support_plan, support_result, rank_input,
        rank_response, bridge, predicate,
    )
    if any(type(item) is not expected for item, expected in zip(values, expected_types)):
        raise TypeError("task decision factory requires exact parent artifact types")
    return (
        ObjectBongardTaskPlan.from_data(task.to_data()),
        ObjectBongardExecutionPrecommit.from_data(precommit.to_data()),
        ObjectSceneAnchorSupportObservationPlan.from_data(support_plan.to_data()),
        ObjectSceneAnchorSupportObservationResult.from_data(support_result.to_data()),
        ObjectSceneAnchorRankInput.from_data(rank_input.to_data()),
        ObjectSceneAnchorRankResponse.from_data(rank_response.to_data()),
        ObjectSceneAnchorPythonBridgeArtifact.from_data(bridge.to_data()),
        ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data()),
    )


def freeze_object_scene_anchor_task_decision(
    *,
    task: ObjectBongardTaskPlan,
    execution_precommit: ObjectBongardExecutionPrecommit,
    support_observation_plan: ObjectSceneAnchorSupportObservationPlan,
    support_observation_result: ObjectSceneAnchorSupportObservationResult,
    rank_input: ObjectSceneAnchorRankInput,
    rank_response: ObjectSceneAnchorRankResponse,
    bridge: ObjectSceneAnchorPythonBridgeArtifact,
    predicate: ObjectSceneAnchorPythonPredicate,
) -> ObjectSceneAnchorTaskDecisionFreeze:
    """Validate all exact parents and freeze one query-independent decision."""

    (
        task, execution_precommit, support_observation_plan,
        support_observation_result, rank_input, rank_response, bridge, predicate,
    ) = _exact_parent_artifacts(
        task, execution_precommit, support_observation_plan,
        support_observation_result, rank_input, rank_response, bridge, predicate,
    )
    expected_support_ids = (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    panels = support_observation_plan.corpus.panels
    if (
        task.task_id not in execution_precommit.selected_task_ids
        or not set(expected_support_ids) <= set(
            execution_precommit.authorized_support_panel_ids
        )
        or not set(query_ids) <= set(execution_precommit.sealed_query_panel_ids)
        or tuple(item.task_id for item in panels) != (task.task_id,) * 12
        or tuple(item.panel_id for item in panels) != expected_support_ids
        or tuple(item.support_bucket_index for item in panels) != (0,) * 6 + (1,) * 6
        or support_observation_result.plan_digest
        != support_observation_plan.plan_digest
        or support_observation_result.corpus_freeze_digest
        != support_observation_plan.corpus.freeze_digest
        or support_observation_result.language_digest
        != support_observation_plan.language.language_digest
        or support_observation_result.source_digest
        != support_observation_plan.source_digest
        or support_observation_result.algorithm_digest
        != support_observation_plan.algorithm_digest
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task, precommit, support panel order, plan, or result binding differs"
        )
    spaces = (
        support_observation_result.bucket0_positive_version_space,
        support_observation_result.bucket1_positive_version_space,
    )
    nonempty = tuple(
        sorted(
            (item for item in spaces if item.survivor_candidate_digests),
            key=lambda item: item.version_space_digest,
        )
    )
    empty = tuple(item for item in spaces if not item.survivor_candidate_digests)
    if not nonempty:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task decision has no nonempty support orientation"
        )
    expected_rank_input = freeze_object_scene_anchor_rank_input(
        nonempty[0], None if len(nonempty) == 1 else nonempty[1]
    )
    if rank_input != expected_rank_input or rank_response.rank_input != rank_input:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "rank input or response is not the exact nonempty orientation union"
        )
    try:
        cold_verify_object_scene_anchor_python_bridge(
            bridge,
            response=rank_response,
            first_version_space=spaces[0],
            second_version_space=spaces[1],
            expected_bridge_digest=bridge.bridge_digest,
            expected_response_digest=rank_response.response_digest,
            expected_rank_input_digest=rank_input.rank_input_digest,
        )
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "bridge does not replay against both exact support orientations"
        ) from exc
    if (
        bridge.predicate != predicate
        or bridge.child_version_space_digests
        != rank_input.child_version_space_digests
        or (len(empty) == 1 and bridge.omitted_version_space != empty[0])
        or (not empty and bridge.omitted_version_space is not None)
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "bridge selection or typed-gap omission proof differs"
        )
    model_artifacts: tuple[object, ...] = (
        support_observation_plan.batch_plan.to_data(),
        rank_response.model_payload,
        rank_response.receipt,
        object_scene_anchor_candidate_ranker_prompt(rank_input),
        *(
            object_scene_anchor_batch_observer_prompt(
                item, support_observation_plan.language.vocabulary
            )
            for item in support_observation_plan.batch_plan.batches
        ),
    )
    if _has_bytes(model_artifacts) or any(
        _contains_text(model_artifacts, query_id) for query_id in query_ids
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "query identity or pixels appear in a model artifact"
        )
    values = {
        "algorithm_digest": object_scene_anchor_task_decision_custody_algorithm_digest(),
        "task_id": task.task_id,
        "task_plan_digest": task.record_digest,
        "execution_precommit_digest": execution_precommit.record_digest,
        "support_observation_plan_digest": support_observation_plan.plan_digest,
        "support_observation_result_digest": support_observation_result.result_digest,
        "support_corpus_freeze_digest": support_observation_plan.corpus.freeze_digest,
        "batch_artifact_digest": support_observation_result.batch_artifact_digest,
        "language_digest": support_observation_plan.language.language_digest,
        "support_panel_ids": expected_support_ids,
        "support_panel_binding_digests": tuple(
            item.source_panel_binding_digest for item in panels
        ),
        "support_panel_png_digests": tuple(
            item.original_panel_png_digest for item in panels
        ),
        "support_evaluation_digests": tuple(
            item.evaluation_digest
            for item in support_observation_result.panel_evaluations
        ),
        "orientation_version_space_digests": tuple(
            item.version_space_digest for item in spaces
        ),
        "version_space_digest": rank_input.version_space_digest,
        "support_version_space_digest": rank_input.version_space_digest,
        "rank_input": rank_input,
        "rank_input_digest": rank_input.rank_input_digest,
        "rank_response": rank_response,
        "rank_response_digest": rank_response.response_digest,
        "bridge": bridge,
        "bridge_digest": bridge.bridge_digest,
        "selected_predicate": predicate,
        "selected_predicate_digest": predicate.predicate_digest,
        "sealed_query_panel_ids": query_ids,
    }
    provisional = object.__new__(ObjectSceneAnchorTaskDecisionFreeze)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorTaskDecisionFreeze(
        **values,
        record_digest=_content_address(_freeze_content(provisional)),
    )


def cold_verify_object_scene_anchor_task_decision_freeze(
    freeze: ObjectSceneAnchorTaskDecisionFreeze,
    *,
    task: ObjectBongardTaskPlan,
    execution_precommit: ObjectBongardExecutionPrecommit,
    support_observation_plan: ObjectSceneAnchorSupportObservationPlan,
    support_observation_result: ObjectSceneAnchorSupportObservationResult,
    rank_input: ObjectSceneAnchorRankInput,
    rank_response: ObjectSceneAnchorRankResponse,
    bridge: ObjectSceneAnchorPythonBridgeArtifact,
    predicate: ObjectSceneAnchorPythonPredicate,
    expected_freeze_digest: str,
) -> ObjectSceneAnchorTaskDecisionFreeze:
    if type(freeze) is not ObjectSceneAnchorTaskDecisionFreeze:
        raise TypeError("freeze has the wrong type")
    restored = ObjectSceneAnchorTaskDecisionFreeze.from_data(freeze.to_data())
    if restored.record_digest != _address(
        expected_freeze_digest, "expected task freeze digest"
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task freeze differs from external commitment"
        )
    expected = freeze_object_scene_anchor_task_decision(
        task=task,
        execution_precommit=execution_precommit,
        support_observation_plan=support_observation_plan,
        support_observation_result=support_observation_result,
        rank_input=rank_input,
        rank_response=rank_response,
        bridge=bridge,
        predicate=predicate,
    )
    if restored != expected:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task decision freeze differs from cold replay"
        )
    return restored


def _commit_content(value: "ObjectSceneAnchorTaskDecisionCommit") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_TASK_DECISION_COMMIT_SCHEMA,
        "custody_id": OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "exact_freeze_payload_size": value.exact_freeze_payload_size,
        "task_freeze_store_receipt": value.task_freeze_store_receipt.to_data(),
        "task_freeze_store_receipt_digest": (
            value.task_freeze_store_receipt_digest
        ),
        "durably_persisted_and_reloaded_before_query_release": True,
        "exact_canonical_freeze_payload_bound": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskDecisionCommit:
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    exact_freeze_payload_size: int
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ObjectSceneAnchorTaskDecisionCustodyError("commit task ID differs")
        for item, label in (
            (self.task_plan_digest, "commit task plan digest"),
            (self.execution_precommit_digest, "commit precommit digest"),
            (self.task_freeze_digest, "commit task freeze digest"),
            (self.exact_freeze_payload_digest, "commit payload digest"),
            (self.task_freeze_store_receipt_digest, "commit receipt digest"),
            (self.record_digest, "task decision commit digest"),
        ):
            _address(item, label)
        for item, label in (
            (self.version_space_digest, "commit version-space digest"),
            (self.support_version_space_digest, "commit support space digest"),
            (self.rank_response_digest, "commit rank response digest"),
            (self.selected_predicate_digest, "commit predicate digest"),
        ):
            _raw_digest(item, label)
        if type(self.task_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("freeze receipt has the wrong type")
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.task_freeze_store_receipt.to_data()
        )
        if (
            type(self.exact_freeze_payload_size) is not int
            or self.exact_freeze_payload_size <= 0
            or self.version_space_digest != self.support_version_space_digest
            or receipt.object_kind != "task-freeze"
            or receipt.object_digest != self.task_freeze_digest
            or receipt.payload_digest != self.exact_freeze_payload_digest
            or receipt.size_bytes != self.exact_freeze_payload_size
            or receipt.record_digest != self.task_freeze_store_receipt_digest
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision durable commit differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorTaskDecisionCommit":
        raw = _fields(
            value,
            {
                "schema", "custody_id", "task_id", "task_plan_digest",
                "execution_precommit_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "exact_freeze_payload_size",
                "task_freeze_store_receipt", "task_freeze_store_receipt_digest",
                "durably_persisted_and_reloaded_before_query_release",
                "exact_canonical_freeze_payload_bound", *_authority_data(),
                "record_digest",
            },
            "anchor task decision commit",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_TASK_DECISION_COMMIT_SCHEMA
            or raw["custody_id"] != OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID
            or raw["durably_persisted_and_reloaded_before_query_release"] is not True
            or raw["exact_canonical_freeze_payload_bound"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["task_freeze_store_receipt"], Mapping)
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision commit policy differs"
            )
        result = cls(
            raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["rank_response_digest"],
            raw["selected_predicate_digest"], raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"], raw["exact_freeze_payload_size"],
            ObjectBongardWriteOnceReceipt.from_data(
                raw["task_freeze_store_receipt"]
            ),
            raw["task_freeze_store_receipt_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision commit is not canonical"
            )
        return result

    def assert_matches(
        self,
        freeze: ObjectSceneAnchorTaskDecisionFreeze,
        exact_freeze_payload: bytes,
    ) -> None:
        if self != commit_object_scene_anchor_task_decision(
            freeze=freeze,
            exact_freeze_payload=exact_freeze_payload,
            task_freeze_store_receipt=self.task_freeze_store_receipt,
        ):
            raise ObjectSceneAnchorTaskDecisionCustodyError(
                "task decision commit differs from cold replay"
            )


def commit_object_scene_anchor_task_decision(
    *,
    freeze: ObjectSceneAnchorTaskDecisionFreeze,
    exact_freeze_payload: bytes,
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt,
) -> ObjectSceneAnchorTaskDecisionCommit:
    """Bind the exact canonical freeze bytes to their durable store receipt."""

    if type(freeze) is not ObjectSceneAnchorTaskDecisionFreeze:
        raise TypeError("freeze has the wrong type")
    frozen = ObjectSceneAnchorTaskDecisionFreeze.from_data(freeze.to_data())
    if type(exact_freeze_payload) is not bytes:
        raise TypeError("exact freeze payload must be bytes")
    expected = canonical_json(frozen.to_data()) + b"\n"
    if exact_freeze_payload != expected:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "freeze payload is not exact canonical JSON"
        )
    if type(task_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("freeze store receipt has the wrong type")
    receipt = ObjectBongardWriteOnceReceipt.from_data(
        task_freeze_store_receipt.to_data()
    )
    payload_digest = _bytes_address(expected)
    if (
        receipt.object_kind != "task-freeze"
        or receipt.object_digest != frozen.record_digest
        or receipt.payload_digest != payload_digest
        or receipt.size_bytes != len(expected)
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "freeze receipt does not bind the exact persisted payload"
        )
    values = {
        "task_id": frozen.task_id,
        "task_plan_digest": frozen.task_plan_digest,
        "execution_precommit_digest": frozen.execution_precommit_digest,
        "version_space_digest": frozen.version_space_digest,
        "support_version_space_digest": frozen.support_version_space_digest,
        "rank_response_digest": frozen.rank_response_digest,
        "selected_predicate_digest": frozen.selected_predicate_digest,
        "task_freeze_digest": frozen.record_digest,
        "exact_freeze_payload_digest": payload_digest,
        "exact_freeze_payload_size": len(expected),
        "task_freeze_store_receipt": receipt,
        "task_freeze_store_receipt_digest": receipt.record_digest,
    }
    provisional = object.__new__(ObjectSceneAnchorTaskDecisionCommit)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorTaskDecisionCommit(
        **values,
        record_digest=_content_address(_commit_content(provisional)),
    )


def cold_verify_object_scene_anchor_task_decision_commit(
    commit: ObjectSceneAnchorTaskDecisionCommit,
    *,
    freeze: ObjectSceneAnchorTaskDecisionFreeze,
    exact_freeze_payload: bytes,
    expected_commit_digest: str,
) -> ObjectSceneAnchorTaskDecisionCommit:
    if type(commit) is not ObjectSceneAnchorTaskDecisionCommit:
        raise TypeError("commit has the wrong type")
    restored = ObjectSceneAnchorTaskDecisionCommit.from_data(commit.to_data())
    if restored.record_digest != _address(
        expected_commit_digest, "expected task commit digest"
    ):
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task decision commit differs from external commitment"
        )
    expected = commit_object_scene_anchor_task_decision(
        freeze=freeze,
        exact_freeze_payload=exact_freeze_payload,
        task_freeze_store_receipt=restored.task_freeze_store_receipt,
    )
    if restored != expected:
        raise ObjectSceneAnchorTaskDecisionCustodyError(
            "task decision commit differs from cold replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_TASK_DECISION_COMMIT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_TASK_DECISION_CUSTODY_ID",
    "OBJECT_SCENE_ANCHOR_TASK_DECISION_FREEZE_SCHEMA",
    "ObjectSceneAnchorTaskDecisionCommit",
    "ObjectSceneAnchorTaskDecisionCustodyError",
    "ObjectSceneAnchorTaskDecisionFreeze",
    "cold_verify_object_scene_anchor_task_decision_commit",
    "cold_verify_object_scene_anchor_task_decision_freeze",
    "commit_object_scene_anchor_task_decision",
    "freeze_object_scene_anchor_task_decision",
    "object_scene_anchor_task_decision_custody_algorithm_digest",
    "object_scene_anchor_task_decision_custody_source_digest",
)
