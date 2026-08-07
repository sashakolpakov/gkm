from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bongard.canonical import canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_codex_ranker import ObjectBongardRankResponse
from bongard.object_bongard_semantics import describe_object_bongard_support
from bongard.object_bongard_release_gate import (
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
)
from bongard.object_bongard_task_runner import (
    ObjectBongardTaskFreeze,
    ObjectBongardTaskFreezeCommit,
    ObjectBongardTaskRunArchive,
    ObjectBongardTaskRunStatus,
    ObjectBongardTaskRunnerError,
    cold_replay_object_bongard_task,
    run_object_bongard_task,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_IDS,
    IntegerInterval,
)
from bongard.prototype_object_version_space import (
    ObjectSceneEvidence,
    ObjectSceneFeatureValue,
    ObjectStableLineageEvidence,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASK_ID = "ff_nact2_5_0042"
PRECOMMIT = "sha256:" + "7" * 64
LINEAGE_CATALOG = "8" * 64
NOMINATED = ("oblique_span_support_ppm", "bird_like_support_ppm")


def _parents():
    plan = ObjectBongardTaskPlan.create(TASK_ID, seed_digest="sha256:" + "6" * 64)
    panel_ids = (*plan.side_0_support_panel_ids, *plan.side_1_support_panel_ids)
    images = {panel_id: _png(index) for index, panel_id in enumerate(panel_ids)}
    payload = {
        "profiles": [
            {
                "group_id": "group_0",
                "rubric": "A pointed bird like form with several slanted spans.",
                "feature_ids": list(NOMINATED),
            },
            {
                "group_id": "group_1",
                "rubric": "A rounded compact contour with a visible opening.",
                "feature_ids": ["rounded_leaf_support_ppm"],
            },
        ]
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    semantic = describe_object_bongard_support(
        task_id=plan.task_id,
        group_0_panel_ids=plan.side_0_support_panel_ids,
        group_1_panel_ids=plan.side_1_support_panel_ids,
        support_png_by_panel_id=images,
        observation_context_digest=PRECOMMIT,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    return plan, semantic


def _scene(scene_id: str, state: str) -> ObjectSceneEvidence:
    values = []
    for feature_id in OBJECT_FEATURE_IDS:
        if feature_id in NOMINATED and state == "uncertain":
            value = ObjectSceneFeatureValue(
                feature_id,
                Disposition.INDETERMINATE,
                None,
                reason="lineage measurement unresolved",
            )
        else:
            level = 1_000_000 if feature_id in NOMINATED and state == "high" else 0
            value = ObjectSceneFeatureValue(
                feature_id, Disposition.PRESENT, IntegerInterval(level, level)
            )
        values.append(value)
    lineage = ObjectStableLineageEvidence.create("lineage-0", values)
    return ObjectSceneEvidence.create(
        scene_id,
        LINEAGE_CATALOG,
        (lineage,),
        unresolved_lineage_possible=False,
    )


def _support(plan, positive_state="high", negative_state="low"):
    return (
        tuple(_scene(panel_id, positive_state) for panel_id in plan.side_0_support_panel_ids),
        tuple(_scene(panel_id, negative_state) for panel_id in plan.side_1_support_panel_ids),
    )


class _Ranker:
    def __init__(self, events):
        self.events = events

    def __call__(self, survivors, **kwargs):
        self.events.append("rank")
        return ObjectBongardRankResponse.seal(
            ordered_profile_digests=[item.profile_digest for item in survivors],
            ranker_protocol_id="test.object-ranker",
            ranker_protocol_digest="1" * 64,
            model_id="test-model",
            model_identity_digest="2" * 64,
            environment_digest="3" * 64,
            rank_input_digest=kwargs["rank_input_digest"],
            transport_receipt={"outcome": "offline-test"},
        )

    def verify_response(self, response, **kwargs):
        self.events.append("verify-rank")
        response.assert_matches(
            survivor_profile_digests=[item.profile_digest for item in kwargs["survivors"]],
            rank_input_digest=kwargs["rank_input_digest"],
        )
        return response


def _run(*, positive_state="high", negative_state="low", wrong_query=False):
    plan, semantic = _parents()
    positives, negatives = _support(plan, positive_state, negative_state)
    events = []
    stored = {}

    def commit(raw):
        events.append("commit")
        stored["freeze"] = raw
        freeze = ObjectBongardTaskFreeze.from_data(json.loads(raw))
        return ObjectBongardTaskFreezeCommit.seal(
            freeze,
            raw,
            task_freeze_store_receipt_digest="sha256:" + "9" * 64,
        )

    def reload(commit_data):
        events.append("reload")
        assert commit_data["task_freeze_store_receipt_digest"] == "sha256:" + "9" * 64
        return stored["freeze"]

    def query(freeze_data, commit_data):
        events.append("query")
        assert freeze_data["query_pixels_included"] is False
        assert commit_data["task_freeze_digest"] == freeze_data["record_digest"]
        side_0_id = "ff/wrong/1/0.png" if wrong_query else plan.side_0_query_panel_id
        return {
            "side_0": _scene(side_0_id, "high"),
            "side_1": _scene(plan.side_1_query_panel_id, "low"),
        }

    archive = run_object_bongard_task(
        plan,
        semantic,
        positives,
        negatives,
        execution_precommit_digest=PRECOMMIT,
        ranker=_Ranker(events),
        freeze_committer=commit,
        freeze_reloader=reload,
        query_source=query,
    )
    return archive, events


def test_complete_run_freezes_durably_before_exact_queries_and_cold_replays() -> None:
    archive, events = _run()
    assert events == ["rank", "verify-rank", "commit", "reload", "query"]
    assert archive.status is ObjectBongardTaskRunStatus.COMPLETE
    assert archive.correct_count == archive.score_denominator == 2
    assert archive.accuracy_ppm == 1_000_000
    assert archive.freeze is not None and archive.freeze_commit is not None
    assert archive.freeze.record_digest.startswith("sha256:")
    assert archive.freeze.support_version_space_digest == archive.freeze.version_space_digest
    assert archive.freeze.selected_predicate_digest == archive.freeze.selected_profile.profile_digest
    assert archive.freeze_commit.task_freeze_digest == archive.freeze.record_digest
    assert isinstance(archive.freeze, ObjectBongardTaskFreezeProtocol)
    assert isinstance(archive.freeze_commit, ObjectBongardTaskCommitProtocol)
    replay = cold_replay_object_bongard_task(
        archive.to_data(), expected_archive_digest=archive.record_digest
    )
    assert replay == archive
    assert replay.to_data()["cold_replay_model_calls"] == 0


@pytest.mark.parametrize(
    ("positive_state", "negative_state", "status"),
    [
        ("low", "high", ObjectBongardTaskRunStatus.LANGUAGE_GAP),
        ("uncertain", "low", ObjectBongardTaskRunStatus.WITNESS_GAP),
    ],
)
def test_typed_gap_never_calls_rank_freeze_or_query(
    positive_state, negative_state, status
) -> None:
    plan, semantic = _parents()
    positives, negatives = _support(plan, positive_state, negative_state)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("later phase crossed")

    archive = run_object_bongard_task(
        plan,
        semantic,
        positives,
        negatives,
        execution_precommit_digest=PRECOMMIT,
        ranker=forbidden,
        freeze_committer=forbidden,
        query_source=forbidden,
    )
    assert archive.status is status
    assert archive.rank_calls_made == archive.freeze_commit_calls_made == 0
    assert archive.query_source_calls_made == 0
    assert cold_replay_object_bongard_task(
        archive, expected_archive_digest=archive.record_digest
    ) == archive


def test_wrong_query_identity_and_archive_tamper_fail_closed() -> None:
    with pytest.raises(ObjectBongardTaskRunnerError, match="sealed identities"):
        _run(wrong_query=True)
    archive, _events = _run()
    changed = deepcopy(archive.to_data())
    changed["correct_count"] = 1
    changed["accuracy_ppm"] = 500_000
    with pytest.raises(ObjectBongardTaskRunnerError):
        ObjectBongardTaskRunArchive.from_data(changed)
    with pytest.raises(ObjectBongardTaskRunnerError, match="external commitment"):
        cold_replay_object_bongard_task(
            archive, expected_archive_digest="0" * 64
        )


def test_freeze_bytes_contain_no_query_evidence_or_pixels() -> None:
    archive, _events = _run()
    assert archive.freeze is not None
    data = json.loads(canonical_json(archive.freeze.to_data()))
    assert data["query_evidence_included"] is False
    assert data["query_pixels_included"] is False
    assert set(data).isdisjoint({"query_evidence", "query_pixels", "exact_png_bytes"})
    assert data["sealed_query_panel_ids"] == list(
        archive.freeze.sealed_query_panel_ids
    )
