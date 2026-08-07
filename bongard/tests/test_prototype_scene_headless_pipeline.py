from __future__ import annotations

import ast
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS
from bongard.prototype_scene_calibration import (
    PrototypeSceneEvaluationContext,
    PrototypeSceneScoreStatus,
    PrototypeSceneTagScore,
    fit_prototype_scene_calibration_family,
)
from bongard.prototype_scene_headless_runner import (
    PrototypeSceneCandidateFreeze,
    PrototypeSceneFreezeCommitReceipt,
    PrototypeSceneHeadlessArchive,
    PrototypeSceneHeadlessError,
    PrototypeSceneHeadlessStatus,
    PrototypeSceneRankResponse,
    cold_replay_prototype_scene_headless_run,
    run_prototype_scene_headless,
)
from bongard.prototype_scene_predicates import (
    PrototypeScenePanelEvaluation,
    PrototypeScenePredicateLibrary,
    PrototypeSceneVerifiedObserverBinding,
)
from bongard.prototype_scene_support_version_space import (
    PrototypeSceneGapKind,
    build_prototype_scene_support_version_space,
    complete_prototype_scene_candidates,
)
from bongard.tests.test_prototype_scene_calibration import (
    _address,
    _calibration_plan,
    _observations,
)
from bongard.triangle_geometry import (
    TriangleGeometryPacket,
    triangle_geometry_algorithm_digest,
)


@pytest.fixture(scope="module")
def scene_authority():
    _cohort, plan = _calibration_plan()
    family = fit_prototype_scene_calibration_family(
        plan,
        _observations(plan),
        expected_calibration_plan_digest=plan.record_digest,
    )
    context = PrototypeSceneEvaluationContext(
        cohort_plan_digest=family.cohort_plan_digest,
        description_catalog_digest=family.description_catalog_digest,
        prototype_reference_digest=family.prototype_reference_digest,
        observer_protocol_id=family.observer_protocol_id,
        observer_protocol_digest=family.observer_protocol_digest,
        model_id=family.model_id,
        model_identity_digest=family.model_identity_digest,
        environment_digest=family.environment_digest,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )
    output = BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return family, PrototypeScenePredicateLibrary.freeze(family), context, output.getvalue()


def _score(tag_id: str, state: str) -> PrototypeSceneTagScore:
    if state == "present":
        return PrototypeSceneTagScore(
            tag_id,
            PrototypeSceneScoreStatus.SCORE,
            800_000,
            900_000,
            "scored",
            None,
        )
    if state == "absent":
        return PrototypeSceneTagScore(
            tag_id,
            PrototypeSceneScoreStatus.SCORE,
            100_000,
            200_000,
            "scored",
            None,
        )
    return PrototypeSceneTagScore(
        tag_id,
        PrototypeSceneScoreStatus.SCORE,
        400_000,
        600_000,
        "scored",
        None,
    )


def _panel(
    authority,
    panel_id: str,
    states: tuple[str, str],
    *,
    geometry: TriangleGeometryPacket | None = None,
) -> PrototypeScenePanelEvaluation:
    family, _library, context, png = authority
    scores = tuple(
        _score(tag_id, state)
        for tag_id, state in zip(OPAQUE_TAG_IDS, states, strict=True)
    )
    binding = PrototypeSceneVerifiedObserverBinding.seal_verified(
        panel_id=panel_id,
        exact_png_bytes=png,
        observer_artifact_schema="gkm.prototype-scene-observer.v1",
        observer_artifact_digest=_address({"artifact": panel_id}),
        verifier_id="prototype.scene.fixture.verifier.v1",
        verifier_digest=_address("fixture-observer-verifier"),
        scores=scores,
        context=context,
    )
    return PrototypeScenePanelEvaluation.seal(
        panel_id=panel_id,
        exact_png_bytes=png,
        observer_binding=binding,
        family=family,
        context=context,
        scores=scores,
        typed_geometry=geometry,
    )


def _conjunction_only_support(authority):
    positives = tuple(
        _panel(authority, f"support-positive-{index}", ("present", "present"))
        for index in range(6)
    )
    negatives = tuple(
        _panel(
            authority,
            f"support-negative-{index}",
            ("absent", "present")
            if index % 2 == 0
            else ("present", "absent"),
        )
        for index in range(6)
    )
    return positives, negatives


def _rank_response(ids: tuple[str, ...], input_digest: str):
    return PrototypeSceneRankResponse.seal(
        ordered_candidate_ids=tuple(reversed(ids)),
        ranker_protocol_id="headless.codex.prototype-scene.v1",
        ranker_protocol_digest=_address("headless-ranker-protocol"),
        model_id="codex-headless-fixture",
        model_identity_digest=_address("codex-model-identity"),
        environment_digest=_address("codex-ranker-environment"),
        input_digest=input_digest,
        receipt={
            "source": "fake-codex-transport",
            "call_count": 1,
            "query_material_included": False,
        },
    )


def _committer(events: list[str]):
    def commit(payload: bytes):
        events.append("commit")
        freeze = PrototypeSceneCandidateFreeze.from_data(json.loads(payload))
        return PrototypeSceneFreezeCommitReceipt.seal(
            freeze, payload, storage_id="fixture-durable-prototype-freeze"
        )

    return commit


def _verifier(events: list[str]):
    def verify(binding, png):
        assert binding.exact_png_digest == (
            "sha256:" + __import__("hashlib").sha256(png).hexdigest()
        )
        assert len(binding.score_digests) == 2
        events.append(f"verify:{binding.panel_id}")

    return verify


def test_finite_library_conjunction_and_nondecisional_geometry(scene_authority) -> None:
    family, library, _context, _png = scene_authority
    assert tuple(item.tag_id for item in library.predicates) == OPAQUE_TAG_IDS
    candidates = complete_prototype_scene_candidates(library)
    assert len(candidates) == 3
    assert sum(len(item.atom_predicate_ids) == 2 for item in candidates) == 1
    geometry = TriangleGeometryPacket(
        panel_digest="1" * 64,
        loop_scene_packet_digest="2" * 64,
        algorithm_digest=triangle_geometry_algorithm_digest(),
        observations=(),
    )
    with_geometry = _panel(
        scene_authority, "geometry-panel", ("present", "absent"), geometry=geometry
    )
    without_geometry = _panel(
        scene_authority, "no-geometry-panel", ("present", "absent")
    )
    assert with_geometry.to_data()["typed_geometry_is_nondecisional"] is True
    assert with_geometry.typed_geometry_digest == geometry.digest
    assert tuple(item.disposition for item in with_geometry.results) == tuple(
        item.disposition for item in without_geometry.results
    )
    assert type(with_geometry).from_data(with_geometry.to_data()) == with_geometry
    with_geometry.assert_matches(family)


def test_complete_run_binds_rank_receipt_and_commit_before_one_plus_one_query(
    scene_authority,
) -> None:
    family, library, _context, _png = scene_authority
    positives, negatives = _conjunction_only_support(scene_authority)
    version = build_prototype_scene_support_version_space(
        library, family, positives, negatives
    )
    assert len(version.survivor_candidate_ids) == 1
    assert version.survivor_candidate_ids[0].startswith(
        "prototype-scene:positive-and:"
    )
    events: list[str] = []

    def ranker(ids, input_digest):
        assert events[-1].startswith("verify:support-negative-")
        events.append("rank")
        return _rank_response(ids, input_digest)

    def query_source(freeze_data):
        assert events[-1] == "commit"
        freeze = PrototypeSceneCandidateFreeze.from_data(freeze_data)
        assert freeze.rank_response_digest.startswith("sha256:")
        events.append("query")
        return {
            "positive": _panel(
                scene_authority, "query-positive", ("present", "present")
            ),
            "negative": _panel(
                scene_authority, "query-negative", ("absent", "present")
            ),
        }

    archive = run_prototype_scene_headless(
        family,
        library,
        positives,
        negatives,
        artifact_verifier=_verifier(events),
        ranker=ranker,
        freeze_committer=_committer(events),
        query_source=query_source,
    )
    assert archive.status is PrototypeSceneHeadlessStatus.COMPLETE
    assert events[12:15] == ["rank", "commit", "query"]
    assert events[-2:] == ["verify:query-positive", "verify:query-negative"]
    assert archive.rank_response is not None and archive.freeze is not None
    assert archive.rank_response.receipt["transport_receipt"]["source"] == (
        "fake-codex-transport"
    )
    assert archive.freeze.rank_response_digest == archive.rank_response.record_digest
    assert archive.freeze_commit is not None
    replay_events: list[str] = []
    replayed = cold_replay_prototype_scene_headless_run(
        archive.to_data(),
        expected_archive_digest=archive.record_digest,
        artifact_verifier=_verifier(replay_events),
    )
    assert replayed.to_data() == archive.to_data()
    assert len(replay_events) == 14
    with pytest.raises(PrototypeSceneHeadlessError, match="external commitment"):
        cold_replay_prototype_scene_headless_run(
            archive,
            expected_archive_digest="sha256:" + "0" * 64,
            artifact_verifier=_verifier([]),
        )


def test_typed_gap_makes_zero_rank_commit_and_query_calls(scene_authority) -> None:
    family, library, _context, _png = scene_authority
    positives = tuple(
        _panel(scene_authority, f"gap-positive-{index}", ("absent", "absent"))
        for index in range(6)
    )
    negatives = tuple(
        _panel(scene_authority, f"gap-negative-{index}", ("present", "present"))
        for index in range(6)
    )
    forbidden_calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        forbidden_calls.append("called")
        raise AssertionError("gap crossed post-version boundary")

    archive = run_prototype_scene_headless(
        family,
        library,
        positives,
        negatives,
        artifact_verifier=_verifier([]),
        ranker=forbidden,
        freeze_committer=forbidden,
        query_source=forbidden,
    )
    assert archive.status is PrototypeSceneHeadlessStatus.LANGUAGE_GAP
    assert archive.version_space.gap is not None
    assert archive.version_space.gap.kind is PrototypeSceneGapKind.LANGUAGE_GAP
    assert forbidden_calls == []
    assert archive.rank_calls_made == archive.query_source_calls_made == 0


def test_bad_rank_commit_or_artifact_verification_blocks_queries(scene_authority) -> None:
    family, library, _context, _png = scene_authority
    positives, negatives = _conjunction_only_support(scene_authority)
    query_calls: list[str] = []

    def bad_rank(ids, input_digest):
        return _rank_response((*ids, "not-a-survivor"), input_digest)

    with pytest.raises(PrototypeSceneHeadlessError, match="exact survivor"):
        run_prototype_scene_headless(
            family,
            library,
            positives,
            negatives,
            artifact_verifier=_verifier([]),
            ranker=bad_rank,
            freeze_committer=lambda _raw: query_calls.append("commit"),
            query_source=lambda _freeze: query_calls.append("query"),
        )
    assert query_calls == []

    with pytest.raises(RuntimeError, match="store unavailable"):
        run_prototype_scene_headless(
            family,
            library,
            positives,
            negatives,
            artifact_verifier=_verifier([]),
            ranker=_rank_response,
            freeze_committer=lambda _raw: (_ for _ in ()).throw(
                RuntimeError("store unavailable")
            ),
            query_source=lambda _freeze: query_calls.append("query"),
        )
    assert query_calls == []

    verifier_calls = 0

    def failed_verifier(_binding, _png):
        nonlocal verifier_calls
        verifier_calls += 1
        raise RuntimeError("observer archive unavailable")

    with pytest.raises(RuntimeError, match="observer archive"):
        run_prototype_scene_headless(
            family,
            library,
            positives,
            negatives,
            artifact_verifier=failed_verifier,
            ranker=_rank_response,
            freeze_committer=_committer([]),
            query_source=lambda _freeze: {},
        )
    assert verifier_calls == 1


def test_raw_score_provenance_rank_receipt_and_authority_tamper_fail_closed(
    scene_authority,
) -> None:
    family, library, _context, _png = scene_authority
    positives, negatives = _conjunction_only_support(scene_authority)
    archive = run_prototype_scene_headless(
        family,
        library,
        positives,
        negatives,
        artifact_verifier=_verifier([]),
        ranker=_rank_response,
        freeze_committer=_committer([]),
        query_source=lambda _freeze: {
            "positive": _panel(
                scene_authority, "tamper-query-positive", ("present", "present")
            ),
            "negative": _panel(
                scene_authority, "tamper-query-negative", ("absent", "present")
            ),
        },
    )
    data = archive.to_data()
    assert data["support_panels"][0]["scores"][0]["lower_ppm"] == 800_000
    assert data["support_panels"][0]["exact_png_base64"]
    assert data["support_panels"][0]["observer_binding"][
        "observer_artifact_verified"
    ] is True
    assert data["rank_response"]["receipt"]
    for record in (
        data,
        data["freeze"],
        data["rank_response"],
        data["support_panels"][0],
    ):
        assert record["python_is_canonical_authority"] is True
        assert record["lean_required"] is False
        assert record["lean_removal_changes_decision"] is False

    for mutate in (
        lambda value: value["support_panels"][0]["scores"][0].__setitem__(
            "lower_ppm", 1
        ),
        lambda value: value["rank_response"]["receipt"][
            "transport_receipt"
        ].__setitem__("source", "forged"),
        lambda value: value.__setitem__("python_is_canonical_authority", False),
    ):
        changed = deepcopy(data)
        mutate(changed)
        with pytest.raises(Exception):
            PrototypeSceneHeadlessArchive.from_data(changed)

    for filename in (
        "prototype_scene_predicates.py",
        "prototype_scene_support_version_space.py",
        "prototype_scene_headless_runner.py",
    ):
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / filename).read_text(
                encoding="utf-8"
            )
        )
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "bongard.multimodal_headless_runner" not in imports
        assert "bongard.multimodal_predicates" not in imports
        assert not any("lean" in item.lower() for item in imports)
