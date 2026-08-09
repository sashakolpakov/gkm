"""Exact rank-union selection and bucket prediction in canonical Python.

The rank response may order survivors, but it cannot edit them.  This module
resolves its first-ranked digest back to the exact child version space, freezes
the existing positive Python predicate, and projects a later P/A/I/E query
decision to one of the two Bongard buckets or to abstention/error.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_candidate_ranker import (
    ObjectSceneAnchorCandidateRankerError,
    ObjectSceneAnchorRankCapacityGap,
    ObjectSceneAnchorRankResponse,
    freeze_object_scene_anchor_rank_input,
    object_scene_anchor_candidate_ranker_protocol_digest,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
    ObjectSceneAnchorSelectionCommitment,
    cold_verify_object_scene_anchor_python_predicate,
    freeze_object_scene_anchor_python_predicate,
    object_scene_anchor_python_predicate_algorithm_digest,
)
from bongard.object_scene_anchor_python_query_observation import (
    ObjectSceneAnchorPythonQueryEvaluation,
    object_scene_anchor_python_query_algorithm_digest,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorSupportVersionSpace,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_SCHEMA = (
    "gkm.object-scene-anchor-python-rank-bridge.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_SCHEMA = (
    "gkm.object-scene-anchor-python-bucket-prediction.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_ALGORITHM_ID = (
    "bongard.object-scene-anchor-python-rank-bridge/exact-two-child-selection-v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_ALGORITHM_ID = (
    "bongard.object-scene-anchor-python-bucket-prediction/four-disposition-v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectSceneAnchorPythonBridgeError(ValueError):
    """An exact rank bridge or deterministic prediction differs."""


class ObjectSceneAnchorPythonBridgeNoResponse(ObjectSceneAnchorPythonBridgeError):
    """No exact rank response exists, so no predicate can be selected."""


class ObjectSceneAnchorPredictionBucket(str, Enum):
    SIDE0_POSITIVE = "side0_positive"
    SIDE1_POSITIVE = "side1_positive"
    ABSTAIN = "abstain"
    ERROR = "error"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "external_inference_calls": 0,
        "ground_truth_material_admitted": False,
        "formal_prover_required": False,
        "polarity_flip_available": False,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorPythonBridgeError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonBridgeError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _orientation(value: object) -> ObjectSceneAnchorOrientation:
    try:
        return ObjectSceneAnchorOrientation(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonBridgeError("bridge orientation differs") from exc


def _disposition(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonBridgeError(
            "prediction disposition differs"
        ) from exc


def _bucket(value: object) -> ObjectSceneAnchorPredictionBucket:
    try:
        return ObjectSceneAnchorPredictionBucket(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonBridgeError(
            "predicted bucket differs"
        ) from exc


def object_scene_anchor_python_bridge_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_python_bridge_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-python-rank-bridge-algorithm.v1",
            "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_ALGORITHM_ID,
            "source_digest": object_scene_anchor_python_bridge_source_digest(),
            "ranker_protocol_digest": (
                object_scene_anchor_candidate_ranker_protocol_digest()
            ),
            "predicate_algorithm_digest": (
                object_scene_anchor_python_predicate_algorithm_digest()
            ),
            "required_child_count": 2,
            "child_spaces_must_be_exact": True,
            "rank_input_must_be_exact_union": True,
            "selected_origin_must_match_candidate": True,
            **_authority_data(),
        }
    )


def object_scene_anchor_python_prediction_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-python-bucket-prediction-algorithm.v1",
            "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_ALGORITHM_ID,
            "source_digest": object_scene_anchor_python_bridge_source_digest(),
            "mapping": {
                Disposition.PRESENT.value: "selected_orientation",
                Disposition.CERTIFIED_ABSENT.value: "opposite_orientation",
                Disposition.INDETERMINATE.value: "abstain",
                Disposition.ERROR.value: "error",
            },
            "orientation_values": [item.value for item in ObjectSceneAnchorOrientation],
            "query_evaluation_algorithm_digest": (
                object_scene_anchor_python_query_algorithm_digest()
            ),
            "preferred_input": "exact-python-query-evaluation",
            "raw_projection_is_lower_level": True,
            **_authority_data(),
        }
    )


def _bridge_content(value: "ObjectSceneAnchorPythonBridgeArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_input_digest": value.rank_input_digest,
        "child_version_space_digests": list(value.child_version_space_digests),
        "child_orientations": [item.value for item in value.child_orientations],
        "selected_origin_version_space_digest": (
            value.selected_origin_version_space_digest
        ),
        "selected_origin_orientation": value.selected_origin_orientation.value,
        "selected_candidate_digest": value.selected_candidate_digest,
        "selection_commitment": value.selection_commitment.to_data(),
        "predicate": value.predicate.to_data(),
        "rank_response_payload_retained": False,
        "both_child_spaces_verified": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonBridgeArtifact:
    """Model-free result of resolving an exact two-child rank response."""

    source_digest: str
    algorithm_digest: str
    rank_response_digest: str
    rank_input_digest: str
    child_version_space_digests: tuple[str, str]
    child_orientations: tuple[
        ObjectSceneAnchorOrientation, ObjectSceneAnchorOrientation
    ]
    selected_origin_version_space_digest: str
    selected_origin_orientation: ObjectSceneAnchorOrientation
    selected_candidate_digest: str
    selection_commitment: ObjectSceneAnchorSelectionCommitment
    predicate: ObjectSceneAnchorPythonPredicate
    bridge_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("bridge source digest", self.source_digest),
            ("bridge algorithm digest", self.algorithm_digest),
            ("rank response digest", self.rank_response_digest),
            ("rank input digest", self.rank_input_digest),
            ("selected version-space digest", self.selected_origin_version_space_digest),
            ("selected candidate digest", self.selected_candidate_digest),
            ("bridge digest", self.bridge_digest),
        ):
            _digest(item, label)
        if (
            type(self.child_version_space_digests) is not tuple
            or len(self.child_version_space_digests) != 2
            or self.child_version_space_digests
            != tuple(sorted(set(self.child_version_space_digests)))
            or type(self.child_orientations) is not tuple
            or len(self.child_orientations) != 2
            or any(
                not isinstance(item, ObjectSceneAnchorOrientation)
                for item in self.child_orientations
            )
            or len(set(self.child_orientations)) != 2
        ):
            raise ObjectSceneAnchorPythonBridgeError(
                "bridge requires two exact distinct child spaces"
            )
        for item in self.child_version_space_digests:
            _digest(item, "child version-space digest")
        if not isinstance(
            self.selected_origin_orientation, ObjectSceneAnchorOrientation
        ):
            raise TypeError("selected bridge orientation has the wrong type")
        if (
            self.selected_origin_version_space_digest,
            self.selected_origin_orientation,
        ) not in set(zip(self.child_version_space_digests, self.child_orientations)):
            raise ObjectSceneAnchorPythonBridgeError(
                "selected origin is outside the exact child inventory"
            )
        if type(self.selection_commitment) is not ObjectSceneAnchorSelectionCommitment:
            raise TypeError("bridge selection commitment has the wrong type")
        if type(self.predicate) is not ObjectSceneAnchorPythonPredicate:
            raise TypeError("bridge predicate has the wrong type")
        selection = ObjectSceneAnchorSelectionCommitment.from_data(
            self.selection_commitment.to_data()
        )
        predicate = ObjectSceneAnchorPythonPredicate.from_data(self.predicate.to_data())
        if (
            self.source_digest != object_scene_anchor_python_bridge_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_bridge_algorithm_digest()
            or selection != self.selection_commitment
            or predicate != self.predicate
            or selection.selection_kind != "exact_rank_response"
            or selection.selector_record_digest != self.rank_response_digest
            or selection.version_space_digest
            != self.selected_origin_version_space_digest
            or selection.orientation is not self.selected_origin_orientation
            or selection.selected_candidate_digest != self.selected_candidate_digest
            or predicate.selection_commitment != selection
            or predicate.version_space_digest
            != self.selected_origin_version_space_digest
            or predicate.candidate.orientation is not self.selected_origin_orientation
            or predicate.candidate.candidate_digest != self.selected_candidate_digest
        ):
            raise ObjectSceneAnchorPythonBridgeError(
                "bridge selection or predicate projection differs"
            )
        if self.bridge_digest != canonical_digest(_bridge_content(self)):
            raise ObjectSceneAnchorPythonBridgeError("bridge digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_bridge_content(self), "bridge_digest": self.bridge_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonBridgeArtifact":
        raw = _exact_fields(
            value,
            {
                "schema", "algorithm_id", "source_digest", "algorithm_digest",
                "rank_response_digest", "rank_input_digest",
                "child_version_space_digests", "child_orientations",
                "selected_origin_version_space_digest", "selected_origin_orientation",
                "selected_candidate_digest", "selection_commitment", "predicate",
                "rank_response_payload_retained", "both_child_spaces_verified",
                *_authority_data(), "bridge_digest",
            },
            "Python rank bridge",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_SCHEMA
            or raw["algorithm_id"] != OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_ALGORITHM_ID
            or raw["rank_response_payload_retained"] is not False
            or raw["both_child_spaces_verified"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["child_version_space_digests"], list)
            or not isinstance(raw["child_orientations"], list)
            or not isinstance(raw["selection_commitment"], Mapping)
            or not isinstance(raw["predicate"], Mapping)
        ):
            raise ObjectSceneAnchorPythonBridgeError("Python rank bridge policy differs")
        child_digests = tuple(raw["child_version_space_digests"])
        child_orientations = tuple(
            _orientation(item) for item in raw["child_orientations"]
        )
        if len(child_digests) != 2 or len(child_orientations) != 2:
            raise ObjectSceneAnchorPythonBridgeError("bridge child count differs")
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            raw["rank_response_digest"],
            raw["rank_input_digest"],
            (child_digests[0], child_digests[1]),
            (child_orientations[0], child_orientations[1]),
            raw["selected_origin_version_space_digest"],
            _orientation(raw["selected_origin_orientation"]),
            raw["selected_candidate_digest"],
            ObjectSceneAnchorSelectionCommitment.from_data(
                raw["selection_commitment"]
            ),
            ObjectSceneAnchorPythonPredicate.from_data(raw["predicate"]),
            raw["bridge_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonBridgeError(
                "Python rank bridge is not canonical"
            )
        return result


def _exact_children(
    first: ObjectSceneAnchorSupportVersionSpace,
    second: ObjectSceneAnchorSupportVersionSpace,
) -> tuple[
    ObjectSceneAnchorSupportVersionSpace,
    ObjectSceneAnchorSupportVersionSpace,
]:
    if type(first) is not ObjectSceneAnchorSupportVersionSpace or type(
        second
    ) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError("bridge child spaces must be exact support version spaces")
    children = tuple(
        sorted(
            (
                ObjectSceneAnchorSupportVersionSpace.from_data(first.to_data()),
                ObjectSceneAnchorSupportVersionSpace.from_data(second.to_data()),
            ),
            key=lambda item: item.version_space_digest,
        )
    )
    return children[0], children[1]


def freeze_object_scene_anchor_python_bridge(
    response: ObjectSceneAnchorRankResponse | None,
    first_version_space: ObjectSceneAnchorSupportVersionSpace,
    second_version_space: ObjectSceneAnchorSupportVersionSpace,
    *,
    expected_response_digest: str,
    expected_rank_input_digest: str,
) -> ObjectSceneAnchorPythonBridgeArtifact:
    """Resolve one exact union response and freeze its selected Python predicate."""

    try:
        rank_input = freeze_object_scene_anchor_rank_input(
            first_version_space, second_version_space
        )
    except ObjectSceneAnchorRankCapacityGap:
        raise
    except ObjectSceneAnchorCandidateRankerError as exc:
        raise ObjectSceneAnchorPythonBridgeError(
            "bridge child spaces do not form one exact rank union"
        ) from exc
    if rank_input.rank_input_digest != _digest(
        expected_rank_input_digest, "expected rank input digest"
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "rank input differs from the exact child-space commitment"
        )
    if response is None:
        raise ObjectSceneAnchorPythonBridgeNoResponse(
            "an exact rank response is required before predicate selection"
        )
    if type(response) is not ObjectSceneAnchorRankResponse:
        raise TypeError("response must be exact ObjectSceneAnchorRankResponse")
    try:
        restored = ObjectSceneAnchorRankResponse.from_data(response.to_data())
    except Exception as exc:
        raise ObjectSceneAnchorPythonBridgeError(
            "rank response failed exact replay"
        ) from exc
    committed_response = _digest(expected_response_digest, "expected rank response digest")
    if (
        restored != response
        or restored.response_digest != committed_response
        or restored.rank_input != rank_input
        or restored.rank_input_digest != rank_input.rank_input_digest
        or restored.child_version_space_digests
        != rank_input.child_version_space_digests
        or restored.child_orientations
        != rank_input.child_orientations
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "rank response differs from both exact child spaces"
        )
    children = _exact_children(first_version_space, second_version_space)
    by_origin = {
        (item.version_space_digest, item.orientation.value): item
        for item in children
    }
    origin_key = (
        restored.selected_origin_version_space_digest,
        restored.selected_origin_orientation,
    )
    try:
        selected_version = by_origin[origin_key]
    except KeyError as exc:
        raise ObjectSceneAnchorPythonBridgeError(
            "rank response selected an unknown child origin"
        ) from exc
    if restored.selected_candidate_digest not in (
        selected_version.survivor_candidate_digests
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "selected candidate is not a survivor of its named child"
        )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        selected_version,
        selected_candidate_digest=restored.selected_candidate_digest,
        selection_kind="exact_rank_response",
        selector_record_digest=restored.response_digest,
    )
    predicate = freeze_object_scene_anchor_python_predicate(
        selected_version, selection
    )
    values = {
        "source_digest": object_scene_anchor_python_bridge_source_digest(),
        "algorithm_digest": object_scene_anchor_python_bridge_algorithm_digest(),
        "rank_response_digest": restored.response_digest,
        "rank_input_digest": rank_input.rank_input_digest,
        "child_version_space_digests": rank_input.child_version_space_digests,
        "child_orientations": tuple(
            ObjectSceneAnchorOrientation(item) for item in rank_input.child_orientations
        ),
        "selected_origin_version_space_digest": selected_version.version_space_digest,
        "selected_origin_orientation": selected_version.orientation,
        "selected_candidate_digest": restored.selected_candidate_digest,
        "selection_commitment": selection,
        "predicate": predicate,
    }
    provisional = object.__new__(ObjectSceneAnchorPythonBridgeArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonBridgeArtifact(
        **values, bridge_digest=canonical_digest(_bridge_content(provisional))
    )


def cold_verify_object_scene_anchor_python_bridge(
    artifact: ObjectSceneAnchorPythonBridgeArtifact,
    *,
    response: ObjectSceneAnchorRankResponse | None,
    first_version_space: ObjectSceneAnchorSupportVersionSpace,
    second_version_space: ObjectSceneAnchorSupportVersionSpace,
    expected_bridge_digest: str,
    expected_response_digest: str,
    expected_rank_input_digest: str,
) -> ObjectSceneAnchorPythonBridgeArtifact:
    """Rebuild the bridge from the rank response and both child spaces."""

    if type(artifact) is not ObjectSceneAnchorPythonBridgeArtifact:
        raise TypeError("artifact must be exact ObjectSceneAnchorPythonBridgeArtifact")
    restored = ObjectSceneAnchorPythonBridgeArtifact.from_data(artifact.to_data())
    if restored.bridge_digest != _digest(
        expected_bridge_digest, "expected bridge digest"
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "bridge differs from external commitment"
        )
    expected = freeze_object_scene_anchor_python_bridge(
        response,
        first_version_space,
        second_version_space,
        expected_response_digest=expected_response_digest,
        expected_rank_input_digest=expected_rank_input_digest,
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonBridgeError("bridge differs from cold replay")
    cold_verify_object_scene_anchor_python_predicate(
        restored.predicate,
        version_space=next(
            item
            for item in _exact_children(first_version_space, second_version_space)
            if item.version_space_digest
            == restored.selected_origin_version_space_digest
        ),
        selection_commitment=restored.selection_commitment,
    )
    return restored


def _opposite_orientation(
    orientation: ObjectSceneAnchorOrientation,
) -> ObjectSceneAnchorOrientation:
    return (
        ObjectSceneAnchorOrientation.SIDE1_POSITIVE
        if orientation is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
        else ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )


def _expected_bucket(
    disposition: Disposition, orientation: ObjectSceneAnchorOrientation
) -> ObjectSceneAnchorPredictionBucket:
    if disposition is Disposition.PRESENT:
        return ObjectSceneAnchorPredictionBucket(orientation.value)
    if disposition is Disposition.CERTIFIED_ABSENT:
        return ObjectSceneAnchorPredictionBucket(_opposite_orientation(orientation).value)
    if disposition is Disposition.INDETERMINATE:
        return ObjectSceneAnchorPredictionBucket.ABSTAIN
    return ObjectSceneAnchorPredictionBucket.ERROR


def _prediction_content(
    value: "ObjectSceneAnchorPythonPrediction",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "predicate_digest": value.predicate_digest,
        "selection_commitment_digest": value.selection_commitment_digest,
        "selected_candidate_digest": value.selected_candidate_digest,
        "predicate_orientation": value.predicate_orientation.value,
        "query_record_digest": value.query_record_digest,
        "query_disposition": value.query_disposition.value,
        "predicted_bucket": value.predicted_bucket.value,
        "mapping_rule": "P:selected,A:opposite,I:abstain,E:error",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonPrediction:
    """Content-addressed bucket projection from one raw query disposition."""

    source_digest: str
    algorithm_digest: str
    predicate_digest: str
    selection_commitment_digest: str
    selected_candidate_digest: str
    predicate_orientation: ObjectSceneAnchorOrientation
    query_record_digest: str
    query_disposition: Disposition
    predicted_bucket: ObjectSceneAnchorPredictionBucket
    prediction_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("prediction source digest", self.source_digest),
            ("prediction algorithm digest", self.algorithm_digest),
            ("prediction predicate digest", self.predicate_digest),
            ("prediction selection digest", self.selection_commitment_digest),
            ("prediction candidate digest", self.selected_candidate_digest),
            ("query record digest", self.query_record_digest),
            ("prediction digest", self.prediction_digest),
        ):
            _digest(item, label)
        if not isinstance(self.predicate_orientation, ObjectSceneAnchorOrientation):
            raise TypeError("prediction orientation has the wrong type")
        if not isinstance(self.query_disposition, Disposition):
            raise TypeError("query disposition has the wrong type")
        if not isinstance(self.predicted_bucket, ObjectSceneAnchorPredictionBucket):
            raise TypeError("predicted bucket has the wrong type")
        if (
            self.source_digest != object_scene_anchor_python_bridge_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_prediction_algorithm_digest()
            or self.predicted_bucket
            is not _expected_bucket(self.query_disposition, self.predicate_orientation)
        ):
            raise ObjectSceneAnchorPythonBridgeError(
                "Python bucket prediction mapping differs"
            )
        if self.prediction_digest != canonical_digest(_prediction_content(self)):
            raise ObjectSceneAnchorPythonBridgeError("prediction digest differs")

    def to_data(self) -> dict[str, object]:
        return {
            **_prediction_content(self),
            "prediction_digest": self.prediction_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonPrediction":
        raw = _exact_fields(
            value,
            {
                "schema", "algorithm_id", "source_digest", "algorithm_digest",
                "predicate_digest", "selection_commitment_digest",
                "selected_candidate_digest", "predicate_orientation",
                "query_record_digest", "query_disposition", "predicted_bucket",
                "mapping_rule", *_authority_data(), "prediction_digest",
            },
            "Python bucket prediction",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_ALGORITHM_ID
            or raw["mapping_rule"] != "P:selected,A:opposite,I:abstain,E:error"
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorPythonBridgeError(
                "Python bucket prediction policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            raw["predicate_digest"],
            raw["selection_commitment_digest"],
            raw["selected_candidate_digest"],
            _orientation(raw["predicate_orientation"]),
            raw["query_record_digest"],
            _disposition(raw["query_disposition"]),
            _bucket(raw["predicted_bucket"]),
            raw["prediction_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonBridgeError(
                "Python bucket prediction is not canonical"
            )
        return result


def project_object_scene_anchor_raw_python_prediction(
    predicate: ObjectSceneAnchorPythonPredicate,
    *,
    query_disposition: Disposition,
    query_record_digest: str,
) -> ObjectSceneAnchorPythonPrediction:
    """Lower-level projection for callers that already bind state to its digest."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if not isinstance(query_disposition, Disposition):
        raise TypeError("query_disposition must be exact Disposition")
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    record = _digest(query_record_digest, "query record digest")
    orientation = frozen.selection_commitment.orientation
    values = {
        "source_digest": object_scene_anchor_python_bridge_source_digest(),
        "algorithm_digest": object_scene_anchor_python_prediction_algorithm_digest(),
        "predicate_digest": frozen.predicate_digest,
        "selection_commitment_digest": (
            frozen.selection_commitment.selection_commitment_digest
        ),
        "selected_candidate_digest": frozen.candidate.candidate_digest,
        "predicate_orientation": orientation,
        "query_record_digest": record,
        "query_disposition": query_disposition,
        "predicted_bucket": _expected_bucket(query_disposition, orientation),
    }
    provisional = object.__new__(ObjectSceneAnchorPythonPrediction)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonPrediction(
        **values,
        prediction_digest=canonical_digest(_prediction_content(provisional)),
    )


def project_object_scene_anchor_python_prediction(
    predicate: ObjectSceneAnchorPythonPredicate,
    evaluation: ObjectSceneAnchorPythonQueryEvaluation,
) -> ObjectSceneAnchorPythonPrediction:
    """Project one exact predicate-bound query evaluation atomically."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(evaluation) is not ObjectSceneAnchorPythonQueryEvaluation:
        raise TypeError(
            "evaluation must be exact ObjectSceneAnchorPythonQueryEvaluation"
        )
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    restored = ObjectSceneAnchorPythonQueryEvaluation.from_data(
        evaluation.to_data()
    )
    if restored.predicate_digest != frozen.predicate_digest:
        raise ObjectSceneAnchorPythonBridgeError(
            "query evaluation belongs to another frozen predicate"
        )
    return project_object_scene_anchor_raw_python_prediction(
        frozen,
        query_disposition=restored.disposition,
        query_record_digest=restored.evaluation_digest,
    )


def cold_verify_object_scene_anchor_raw_python_prediction(
    prediction: ObjectSceneAnchorPythonPrediction,
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    query_disposition: Disposition,
    query_record_digest: str,
    expected_prediction_digest: str,
) -> ObjectSceneAnchorPythonPrediction:
    """Replay the lower-level raw four-way bucket projection."""

    if type(prediction) is not ObjectSceneAnchorPythonPrediction:
        raise TypeError("prediction must be exact ObjectSceneAnchorPythonPrediction")
    restored = ObjectSceneAnchorPythonPrediction.from_data(prediction.to_data())
    if restored.prediction_digest != _digest(
        expected_prediction_digest, "expected prediction digest"
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "prediction differs from external commitment"
        )
    expected = project_object_scene_anchor_raw_python_prediction(
        predicate,
        query_disposition=query_disposition,
        query_record_digest=query_record_digest,
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonBridgeError(
            "prediction differs from cold replay"
        )
    return restored


def cold_verify_object_scene_anchor_python_prediction(
    prediction: ObjectSceneAnchorPythonPrediction,
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    evaluation: ObjectSceneAnchorPythonQueryEvaluation,
    expected_prediction_digest: str,
) -> ObjectSceneAnchorPythonPrediction:
    """Replay a prediction from its exact predicate-bound query evaluation."""

    if type(prediction) is not ObjectSceneAnchorPythonPrediction:
        raise TypeError("prediction must be exact ObjectSceneAnchorPythonPrediction")
    restored = ObjectSceneAnchorPythonPrediction.from_data(prediction.to_data())
    if restored.prediction_digest != _digest(
        expected_prediction_digest, "expected prediction digest"
    ):
        raise ObjectSceneAnchorPythonBridgeError(
            "prediction differs from external commitment"
        )
    expected = project_object_scene_anchor_python_prediction(predicate, evaluation)
    if restored != expected:
        raise ObjectSceneAnchorPythonBridgeError(
            "prediction differs from exact query-evaluation replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_PYTHON_BRIDGE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_PYTHON_PREDICTION_SCHEMA",
    "ObjectSceneAnchorPredictionBucket",
    "ObjectSceneAnchorPythonBridgeArtifact",
    "ObjectSceneAnchorPythonBridgeError",
    "ObjectSceneAnchorPythonBridgeNoResponse",
    "ObjectSceneAnchorPythonPrediction",
    "cold_verify_object_scene_anchor_python_bridge",
    "cold_verify_object_scene_anchor_python_prediction",
    "cold_verify_object_scene_anchor_raw_python_prediction",
    "freeze_object_scene_anchor_python_bridge",
    "object_scene_anchor_python_bridge_algorithm_digest",
    "object_scene_anchor_python_bridge_source_digest",
    "object_scene_anchor_python_prediction_algorithm_digest",
    "project_object_scene_anchor_python_prediction",
    "project_object_scene_anchor_raw_python_prediction",
)
