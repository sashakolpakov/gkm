"""Offline tests for exact rank-union to Python-predicate selection."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json

import pytest

import bongard.object_scene_anchor_candidate_ranker as ranker_module
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_candidate_ranker import (
    ObjectSceneAnchorRankCapacityGap,
    freeze_object_scene_anchor_rank_input,
)
from bongard.object_scene_anchor_python_bridge import (
    ObjectSceneAnchorPredictionBucket,
    ObjectSceneAnchorPythonBridgeArtifact,
    ObjectSceneAnchorPythonBridgeError,
    ObjectSceneAnchorPythonBridgeNoResponse,
    ObjectSceneAnchorPythonPrediction,
    cold_verify_object_scene_anchor_python_bridge,
    cold_verify_object_scene_anchor_python_prediction,
    cold_verify_object_scene_anchor_raw_python_prediction,
    freeze_object_scene_anchor_python_bridge,
    project_object_scene_anchor_python_prediction,
    project_object_scene_anchor_raw_python_prediction,
)
from bongard.object_scene_anchor_python_predicate import (
    cold_verify_object_scene_anchor_python_predicate,
)
from bongard.object_scene_anchor_python_query_observation import (
    ObjectSceneAnchorPythonQueryEvaluation,
    _evaluation_content,
    object_scene_anchor_python_query_algorithm_digest,
)
from bongard.object_scene_anchor_version_space import ObjectSceneAnchorOrientation
from bongard.tests.test_object_scene_anchor_candidate_ranker import (
    _Transport,
    _dual_versions,
    _ranker,
    _version,
)


@lru_cache(maxsize=1)
def _bridge_fixture():
    side0, side1 = _dual_versions()
    rank_input = freeze_object_scene_anchor_rank_input(side0, side1)
    transport = _Transport()
    response = _ranker(transport)(
        side0,
        side1,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    assert transport.calls == 1
    bridge = freeze_object_scene_anchor_python_bridge(
        response,
        side0,
        side1,
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    return side0, side1, rank_input, response, bridge


def _opposite(orientation: ObjectSceneAnchorOrientation) -> ObjectSceneAnchorOrientation:
    return (
        ObjectSceneAnchorOrientation.SIDE1_POSITIVE
        if orientation is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
        else ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )


def _query_evaluation(
    predicate,
    disposition: Disposition,
    *,
    index: int = 0,
    predicate_digest: str | None = None,
) -> ObjectSceneAnchorPythonQueryEvaluation:
    values = {
        "algorithm_digest": object_scene_anchor_python_query_algorithm_digest(),
        "predicate_digest": (
            predicate.predicate_digest
            if predicate_digest is None
            else predicate_digest
        ),
        "observation_digest": canonical_digest(
            {"schema": "test.query-observation.v1", "index": index}
        ),
        "panel_id": f"bridge_query_{index:02d}",
        "disposition": disposition,
    }
    provisional = object.__new__(ObjectSceneAnchorPythonQueryEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonQueryEvaluation(
        **values,
        evaluation_digest=canonical_digest(_evaluation_content(provisional)),
    )


def test_union_response_selects_exact_child_and_freezes_python_predicate() -> None:
    side0, side1, rank_input, response, bridge = _bridge_fixture()
    selected_version = next(
        item
        for item in (side0, side1)
        if item.version_space_digest
        == response.selected_origin_version_space_digest
    )

    assert bridge.rank_response_digest == response.response_digest
    assert bridge.rank_input_digest == rank_input.rank_input_digest
    assert bridge.child_version_space_digests == rank_input.child_version_space_digests
    assert tuple(item.value for item in bridge.child_orientations) == (
        rank_input.child_orientations
    )
    assert bridge.selected_origin_version_space_digest == (
        selected_version.version_space_digest
    )
    assert bridge.selected_origin_orientation is selected_version.orientation
    assert bridge.selected_candidate_digest in (
        selected_version.survivor_candidate_digests
    )
    assert bridge.selection_commitment.selection_kind == "exact_rank_response"
    assert bridge.selection_commitment.selector_record_digest == response.response_digest
    assert bridge.predicate.selection_commitment == bridge.selection_commitment
    assert bridge.predicate.version_space_digest == selected_version.version_space_digest
    assert bridge.predicate.candidate.candidate_digest == response.selected_candidate_digest

    assert ObjectSceneAnchorPythonBridgeArtifact.from_data(bridge.to_data()) == bridge
    assert cold_verify_object_scene_anchor_python_predicate(
        bridge.predicate,
        version_space=selected_version,
        selection_commitment=bridge.selection_commitment,
    ) == bridge.predicate
    assert cold_verify_object_scene_anchor_python_bridge(
        bridge,
        response=response,
        first_version_space=side1,
        second_version_space=side0,
        expected_bridge_digest=bridge.bridge_digest,
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    ) == bridge

    encoded = json.dumps(bridge.to_data(), sort_keys=True).casefold()
    assert "rank_response_payload" in encoded
    assert '"rank_response":' not in encoded


def test_bridge_rejects_mismatch_resealed_tamper_and_no_response() -> None:
    side0, side1, rank_input, response, bridge = _bridge_fixture()
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="external commitment"):
        cold_verify_object_scene_anchor_python_bridge(
            bridge,
            response=response,
            first_version_space=side0,
            second_version_space=side1,
            expected_bridge_digest="0" * 64,
            expected_response_digest=response.response_digest,
            expected_rank_input_digest=rank_input.rank_input_digest,
        )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="child-space"):
        freeze_object_scene_anchor_python_bridge(
            response,
            side0,
            side1,
            expected_response_digest=response.response_digest,
            expected_rank_input_digest="1" * 64,
        )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="rank union"):
        freeze_object_scene_anchor_python_bridge(
            response,
            _version(),
            side1,
            expected_response_digest=response.response_digest,
            expected_rank_input_digest=rank_input.rank_input_digest,
        )
    with pytest.raises(ObjectSceneAnchorPythonBridgeNoResponse, match="required"):
        freeze_object_scene_anchor_python_bridge(
            None,
            side0,
            side1,
            expected_response_digest=response.response_digest,
            expected_rank_input_digest=rank_input.rank_input_digest,
        )

    tampered = deepcopy(bridge.to_data())
    tampered["selected_origin_orientation"] = _opposite(
        bridge.selected_origin_orientation
    ).value
    tampered["bridge_digest"] = canonical_digest(
        {key: value for key, value in tampered.items() if key != "bridge_digest"}
    )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="selected origin"):
        ObjectSceneAnchorPythonBridgeArtifact.from_data(tampered)


def test_capacity_gap_occurs_before_response_use_or_any_call(monkeypatch) -> None:
    side0, side1, rank_input, response, _bridge = _bridge_fixture()
    monkeypatch.setattr(ranker_module, "MAX_SURVIVOR_COUNT", 13)
    with pytest.raises(ObjectSceneAnchorRankCapacityGap) as caught:
        freeze_object_scene_anchor_python_bridge(
            response,
            side0,
            side1,
            expected_response_digest=response.response_digest,
            expected_rank_input_digest=rank_input.rank_input_digest,
        )
    assert caught.value.survivor_count == 14
    assert caught.value.maximum_survivor_count == 13


@pytest.mark.parametrize(
    ("disposition", "expected"),
    (
        (Disposition.PRESENT, "selected"),
        (Disposition.CERTIFIED_ABSENT, "opposite"),
        (Disposition.INDETERMINATE, "abstain"),
        (Disposition.ERROR, "error"),
    ),
)
def test_four_dispositions_project_to_exact_bucket_and_cold_replay(
    disposition: Disposition,
    expected: str,
) -> None:
    bridge = _bridge_fixture()[-1]
    orientation = bridge.selected_origin_orientation
    evaluation = _query_evaluation(
        bridge.predicate,
        disposition,
        index=tuple(Disposition).index(disposition),
    )
    prediction = project_object_scene_anchor_python_prediction(
        bridge.predicate, evaluation
    )
    expected_bucket = {
        "selected": ObjectSceneAnchorPredictionBucket(orientation.value),
        "opposite": ObjectSceneAnchorPredictionBucket(_opposite(orientation).value),
        "abstain": ObjectSceneAnchorPredictionBucket.ABSTAIN,
        "error": ObjectSceneAnchorPredictionBucket.ERROR,
    }[expected]
    assert prediction.query_disposition is disposition
    assert prediction.predicted_bucket is expected_bucket
    assert ObjectSceneAnchorPythonPrediction.from_data(prediction.to_data()) == prediction
    assert cold_verify_object_scene_anchor_python_prediction(
        prediction,
        predicate=bridge.predicate,
        evaluation=evaluation,
        expected_prediction_digest=prediction.prediction_digest,
    ) == prediction
    assert project_object_scene_anchor_raw_python_prediction(
        bridge.predicate,
        query_disposition=evaluation.disposition,
        query_record_digest=evaluation.evaluation_digest,
    ) == prediction

    encoded = json.dumps(prediction.to_data(), sort_keys=True).casefold()
    assert "label" not in encoded
    assert "model" not in encoded
    assert "lean" not in encoded


def test_prediction_resealed_bucket_tamper_and_replay_mismatch_fail_closed() -> None:
    bridge = _bridge_fixture()[-1]
    present = _query_evaluation(bridge.predicate, Disposition.PRESENT, index=7)
    prediction = project_object_scene_anchor_python_prediction(
        bridge.predicate, present
    )
    tampered = deepcopy(prediction.to_data())
    tampered["predicted_bucket"] = ObjectSceneAnchorPredictionBucket.ABSTAIN.value
    tampered["prediction_digest"] = canonical_digest(
        {key: value for key, value in tampered.items() if key != "prediction_digest"}
    )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="mapping differs"):
        ObjectSceneAnchorPythonPrediction.from_data(tampered)

    absent = _query_evaluation(
        bridge.predicate, Disposition.CERTIFIED_ABSENT, index=8
    )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="query-evaluation replay"):
        cold_verify_object_scene_anchor_python_prediction(
            prediction,
            predicate=bridge.predicate,
            evaluation=absent,
            expected_prediction_digest=prediction.prediction_digest,
        )

    foreign = _query_evaluation(
        bridge.predicate,
        Disposition.PRESENT,
        index=9,
        predicate_digest="f" * 64,
    )
    with pytest.raises(ObjectSceneAnchorPythonBridgeError, match="another frozen"):
        project_object_scene_anchor_python_prediction(bridge.predicate, foreign)

    assert cold_verify_object_scene_anchor_raw_python_prediction(
        prediction,
        predicate=bridge.predicate,
        query_disposition=present.disposition,
        query_record_digest=present.evaluation_digest,
        expected_prediction_digest=prediction.prediction_digest,
    ) == prediction
