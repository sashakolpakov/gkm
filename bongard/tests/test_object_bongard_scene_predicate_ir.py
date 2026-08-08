from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO

import pytest
from PIL import Image

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
import bongard.object_bongard_scene_predicate_ir as ir
from bongard.object_bongard_scene_predicate_ir import (
    ObjectBongardScenePredicateIRError,
    SceneAtomKind,
    SceneComparison,
    SceneEntityObservation,
    SceneFormulaNode,
    SceneMergedCell,
    SceneNumericInterval,
    SceneNumericUnit,
    SceneOrientation,
    ScenePanelObservation,
    SceneQuantifier,
    SceneScope,
    adapt_object_scene_registered_pair,
    adapt_object_scene_registered_single,
    build_object_bongard_scene_predicate_calibration_bundle,
    cold_replay_object_bongard_scene_predicate_calibration_bundle,
    enumerate_object_scene_candidates,
    evaluate_object_scene_candidate,
    freeze_object_scene_predicate_language,
    merge_repeated_disposition,
    merge_repeated_interval,
    scene_and,
    scene_or,
)
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_COUNT_OBSERVABLE_IDS,
    OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS,
    ObjectSceneTranscriptMode,
    extract_object_scene_proposal_inventory,
    freeze_object_scene_soft_tag_registry,
    observe_object_scene_transcript,
)
from bongard.tests.test_object_scene_visual_frontend import _payload, _scene, _transport
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
)


def _raw_digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _address(label: str) -> str:
    return "sha256:" + _raw_digest(label)


def _entity(bird: Disposition, *, object_id: str = "object_0000") -> SceneEntityObservation:
    qualitative = tuple(
        SceneMergedCell(
            observable,
            bird if observable == "bird_like" else Disposition.CERTIFIED_ABSENT,
            None,
            (),
        )
        for observable in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS
    )
    counts = tuple(
        SceneMergedCell(observable, Disposition.INDETERMINATE, None, ())
        for observable in OBJECT_SCENE_COUNT_OBSERVABLE_IDS
    )
    return SceneEntityObservation(
        object_id,
        _raw_digest("crop-" + object_id),
        (1000, 1000, 9000, 7000),
        80,
        1,
        0,
        (),
        qualitative,
        counts,
        (),
    )


def _panel(
    panel_id: str,
    bird: Disposition,
    *,
    mode: str = "repeated_registered_merge",
    empty: bool = False,
) -> ScenePanelObservation:
    sources = (_raw_digest(panel_id + "-a"),) if mode == "single_registered" else tuple(
        sorted((_raw_digest(panel_id + "-a"), _raw_digest(panel_id + "-b")))
    )
    values = {
        "panel_id": panel_id,
        "panel_digest": _raw_digest(panel_id + "-png"),
        "inventory_digest": _raw_digest(panel_id + "-inventory"),
        "registry_digest": freeze_object_scene_soft_tag_registry(()).registry_digest,
        "observation_mode": mode,
        "source_artifact_digests": sources,
        "source_transcript_digests": (),
        "disposition": Disposition.PRESENT,
        "entities": () if empty else (_entity(bird),),
    }
    provisional = object.__new__(ScenePanelObservation)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ScenePanelObservation(
        **values,
        observation_digest=canonical_digest(ir._observation_content(provisional)),
    )


def _observe(raw, payload, *, scene_id: str, context: str, mode, registry=None):
    return observe_object_scene_transcript(
        raw,
        scene_id=scene_id,
        observation_context_digest=_address(context),
        mode=mode,
        registry=registry,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=_transport(payload, []),
    )


def _blank_scene() -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    output = BytesIO(); image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _artifact_fixture(*, flip_group0_b: bool = False):
    raws = (_scene(0), _scene(2))
    inventories = tuple(extract_object_scene_proposal_inventory(raw) for raw in raws)
    discovery = tuple(
        _observe(
            raw,
            _payload(inventory, open_tags=("bird-like object",)),
            scene_id=f"calibration_panel_{index:02d}",
            context=f"discovery-{index}",
            mode=ObjectSceneTranscriptMode.DISCOVERY,
        )
        for index, (raw, inventory) in enumerate(zip(raws, inventories, strict=True))
    )
    registry = freeze_object_scene_soft_tag_registry(tuple(item.transcript for item in discovery))

    def registered_payload(index: int, *, flip: bool):
        payload = _payload(inventories[index], registry=registry)
        state = "present" if index == 0 and not flip else "absent"
        for row in payload["objects"]:
            for cell in row["observables"]:
                if cell["observable_id"] == "bird_like":
                    cell["state"] = state
                    cell["evidence"] = "bird silhouette visibly supported" if state == "present" else "bird silhouette is not visible"
            for cell in row["registered_tags"]:
                cell["state"] = state
                cell["evidence"] = "bird silhouette visibly supported" if state == "present" else "bird silhouette is not visible"
        return payload

    pass_a = tuple(
        _observe(raw, registered_payload(index, flip=False), scene_id=f"calibration_panel_{index:02d}", context=f"pass-a-{index}", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
        for index, raw in enumerate(raws)
    )
    pass_b = tuple(
        _observe(raw, registered_payload(index, flip=flip_group0_b and index == 0), scene_id=f"calibration_panel_{index:02d}", context=f"pass-b-{index}", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
        for index, raw in enumerate(raws)
    )
    roles = tuple(
        {
            "ordinal": index,
            "neutral_panel_digest": _raw_digest(f"neutral-{index}"),
            "historical_role": index,
            "blind_panel_id": f"calibration_panel_{index:02d}",
        }
        for index in range(2)
    )
    return registry, discovery, pass_a, pass_b, roles


def _bird_candidate(language, orientation=SceneOrientation.GROUP0_POSITIVE):
    return next(
        candidate
        for candidate in enumerate_object_scene_candidates(language)
        if candidate.orientation is orientation
        and candidate.formula.node is SceneFormulaNode.QUANTIFIED
        and candidate.formula.quantifier is SceneQuantifier.EXISTS
        and candidate.formula.children[0].atom is not None
        and candidate.formula.children[0].atom.kind is SceneAtomKind.QUALITATIVE
        and candidate.formula.children[0].atom.observable_id == "bird_like"
    )


def test_four_state_merge_interval_and_error_dominance():
    assert merge_repeated_disposition(Disposition.PRESENT, Disposition.PRESENT) is Disposition.PRESENT
    assert merge_repeated_disposition(Disposition.CERTIFIED_ABSENT, Disposition.CERTIFIED_ABSENT) is Disposition.CERTIFIED_ABSENT
    assert merge_repeated_disposition(Disposition.PRESENT, Disposition.CERTIFIED_ABSENT) is Disposition.INDETERMINATE
    assert merge_repeated_disposition(None, Disposition.CERTIFIED_ABSENT) is Disposition.ERROR
    assert scene_and((Disposition.ERROR, Disposition.CERTIFIED_ABSENT)) is Disposition.ERROR
    assert scene_or((Disposition.ERROR, Disposition.PRESENT)) is Disposition.ERROR
    state, interval = merge_repeated_interval(
        SceneNumericInterval(SceneNumericUnit.COUNT, 1, 4),
        SceneNumericInterval(SceneNumericUnit.COUNT, 3, 5),
    )
    assert state is Disposition.PRESENT and interval == SceneNumericInterval(SceneNumericUnit.COUNT, 3, 4)
    assert merge_repeated_interval(SceneNumericInterval(SceneNumericUnit.COUNT, 0, 1), SceneNumericInterval(SceneNumericUnit.COUNT, 2, 3)) == (Disposition.INDETERMINATE, None)


def test_positive_closed_language_both_orientations_registry_binding_and_empty_all():
    registry = freeze_object_scene_soft_tag_registry(())
    group0, group1 = _panel("group0", Disposition.PRESENT), _panel("group1", Disposition.CERTIFIED_ABSENT)
    language = freeze_object_scene_predicate_language(registry, (group0, group1))
    candidates = enumerate_object_scene_candidates(language)
    bird0 = _bird_candidate(language)
    assert evaluate_object_scene_candidate(bird0, language, group0) is Disposition.PRESENT
    assert evaluate_object_scene_candidate(bird0, language, group1) is Disposition.CERTIFIED_ABSENT
    assert {item.orientation for item in candidates} == set(SceneOrientation)
    assert len(candidates) > ir.SCENE_MAX_RANK_SLATE
    selected = ir._semantically_stratified_rank_selection(candidates)
    all_families = {ir._candidate_rank_family(item) for item in candidates}
    assert {ir._candidate_rank_family(item) for item in selected} == all_families
    assert len({ir._candidate_rank_family(item) for item in selected[: len(all_families)]}) == len(all_families)
    for family in all_families:
        family_selected = [item for item in selected if ir._candidate_rank_family(item) == family]
        all_strata = {ir._candidate_rank_stratum(item) for item in candidates if ir._candidate_rank_family(item) == family}
        seen = set()
        for item in family_selected:
            stratum = ir._candidate_rank_stratum(item)
            if stratum in seen:
                assert seen == all_strata
            seen.add(stratum)
    assert all(item.formula.node is not getattr(SceneFormulaNode, "OR", None) for item in candidates)
    for candidate in candidates:
        stack = [candidate.formula]
        while stack:
            formula = stack.pop(); stack.extend(formula.children)
            boundary_id = formula.count_boundary_id or (None if formula.atom is None else formula.atom.boundary_id)
            if boundary_id is not None:
                boundary = language.boundary(boundary_id)
                assert boundary.value >= 1
                assert boundary.comparison is not SceneComparison.AT_MOST
    all_bird = next(item for item in candidates if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.quantifier is SceneQuantifier.ALL and item.formula.children[0].atom is not None and item.formula.children[0].atom.observable_id == "bird_like")
    assert evaluate_object_scene_candidate(all_bird, language, _panel("empty", Disposition.INDETERMINATE, empty=True)) is Disposition.INDETERMINATE
    with pytest.raises(ObjectBongardScenePredicateIRError, match="different soft-tag registry"):
        wrong = deepcopy(group0.to_data()); wrong["registry_digest"] = _raw_digest("wrong-registry"); wrong["observation_digest"] = canonical_digest({key: value for key, value in wrong.items() if key != "observation_digest"})
        evaluate_object_scene_candidate(bird0, language, ScenePanelObservation.from_data(wrong))
    with pytest.raises(ObjectBongardScenePredicateIRError, match="repeated observations"):
        freeze_object_scene_predicate_language(registry, (_panel("single", Disposition.PRESENT, mode="single_registered"), group1))


def test_distinct_calls_required_and_real_pass_b_disagreement_fails_repeatability():
    registry, discovery, pass_a, pass_b, roles = _artifact_fixture(flip_group0_b=True)
    with pytest.raises(ObjectBongardScenePredicateIRError, match="two distinct"):
        adapt_object_scene_registered_pair("calibration_panel_00", pass_a[0], pass_a[0])
    bundle = build_object_bongard_scene_predicate_calibration_bundle(registry, discovery, pass_a, pass_b, roles)
    assert bundle.coverage_gate.passed is True
    assert bundle.selectivity_gate.passed is True
    assert bundle.repeatability_gate.passed is False
    assert bundle.complete_survivor_digests == ()
    assert bundle.version_space["orientation_spaces"][0]["pass_a_evaluations"]
    assert bundle.version_space["orientation_spaces"][0]["pass_b_evaluations"]


def test_zero_proposal_artifacts_are_error_and_one_entity_pair_quantifiers_are_nonvacuous():
    raw = _blank_scene(); inventory = extract_object_scene_proposal_inventory(raw)
    assert inventory.objects == ()
    registry = freeze_object_scene_soft_tag_registry(())
    first = _observe(raw, {"objects": []}, scene_id="blank_panel", context="blank-a", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
    second = _observe(raw, {"objects": []}, scene_id="blank_panel", context="blank-b", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
    merged = adapt_object_scene_registered_pair("blank_panel", first, second)
    single = adapt_object_scene_registered_single("blank_panel", first)
    assert merged.disposition is single.disposition is Disposition.ERROR
    language = freeze_object_scene_predicate_language(registry, (_panel("group0z", Disposition.PRESENT), _panel("group1z", Disposition.CERTIFIED_ABSENT)))
    assert evaluate_object_scene_candidate(_bird_candidate(language), language, merged) is Disposition.ERROR
    pair_exists = next(item for item in enumerate_object_scene_candidates(language) if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.scope is SceneScope.PAIR and item.formula.quantifier is SceneQuantifier.EXISTS)
    pair_all = next(item for item in enumerate_object_scene_candidates(language) if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.scope is SceneScope.PAIR and item.formula.quantifier is SceneQuantifier.ALL)
    one = _panel("one_entity", Disposition.PRESENT)
    assert evaluate_object_scene_candidate(pair_exists, language, one) is Disposition.CERTIFIED_ABSENT
    assert evaluate_object_scene_candidate(pair_all, language, one) is Disposition.INDETERMINATE


def test_bundle_round_trip_cold_replay_registry_provenance_capacity_and_stratified_slate(monkeypatch):
    registry, discovery, pass_a, pass_b, roles = _artifact_fixture()
    bundle = build_object_bongard_scene_predicate_calibration_bundle(registry, discovery, pass_a, pass_b, roles)
    assert len(bundle.candidates) == 1888
    assert len(canonical_json(bundle.to_data())) < 16 * 1024 * 1024
    restored = ir.ScenePredicateCalibrationBundle.from_data(bundle.to_data())
    assert cold_replay_object_bongard_scene_predicate_calibration_bundle(restored, registry) == bundle
    assert any("resembles a bird or flying bird silhouette" in str(item) for item in bundle.ranker_slate)
    if len(bundle.complete_survivor_digests) > 64:
        complexities = {item["complexity"] for item in bundle.ranker_slate}
        survivor_complexities = {item.complexity for item in bundle.candidates if item.candidate_digest in bundle.complete_survivor_digests}
        assert complexities == survivor_complexities or len(complexities) >= 2
    tampered = deepcopy(bundle.to_data()); tampered["ranker_slate"] = []
    with pytest.raises(ObjectBongardScenePredicateIRError):
        ir.ScenePredicateCalibrationBundle.from_data(tampered)
    language = ir.ScenePredicateLanguage.from_data(bundle.version_space["language"])
    forbidden_boundary = next(item for item in language.boundaries if item.observable_id == "matching_entity_count" and item.comparison is SceneComparison.AT_MOST and item.value >= 1)
    body = ir.SceneFormula.atom_formula(ir.SceneAtom.create(SceneScope.ENTITY, SceneAtomKind.QUALITATIVE, "bird_like"))
    formula = ir.SceneFormula.quantified(SceneScope.ENTITY, SceneQuantifier.COUNT, body, forbidden_boundary.boundary_id)
    values = {"language_digest": language.language_digest, "orientation": SceneOrientation.GROUP0_POSITIVE, "formula": formula, "complexity": formula.complexity}
    provisional = object.__new__(ir.ScenePredicateCandidate)
    for key, value in values.items(): object.__setattr__(provisional, key, value)
    forbidden = ir.ScenePredicateCandidate(**values, candidate_digest=canonical_digest(ir._candidate_content(provisional)))
    self_digested_forbidden = deepcopy(bundle.to_data()); self_digested_forbidden["candidates"][0] = forbidden.to_data()
    with pytest.raises(ObjectBongardScenePredicateIRError, match="COUNT cannot encode absence"):
        ir.ScenePredicateCalibrationBundle.from_data(self_digested_forbidden)
    with pytest.raises(Exception):
        build_object_bongard_scene_predicate_calibration_bundle(freeze_object_scene_soft_tag_registry(()), discovery, pass_a, pass_b, roles)
    monkeypatch.setattr(ir, "SCENE_MAX_ENUMERATED_FORMULAS", 1)
    capacity = build_object_bongard_scene_predicate_calibration_bundle(registry, discovery, pass_a, pass_b, roles)
    assert capacity.complete_survivor_digests == () and capacity.ranker_slate == ()
    assert capacity.version_space["resource_gap"]["reason"] == "complete_formula_inventory_exceeds_resource_cap"
    assert capacity.version_space["full_candidate_space_persisted"] is False
    assert cold_replay_object_bongard_scene_predicate_calibration_bundle(capacity, registry) == capacity
    fake_complete = deepcopy(capacity.to_data())
    fake_complete["version_space"]["resource_gap"] = None
    fake_complete["version_space"]["full_candidate_space_persisted"] = True
    fake_complete["version_space"]["complete_space_accounted_by_typed_capacity_gap"] = False
    with pytest.raises(ObjectBongardScenePredicateIRError, match="capacity accounting"):
        ir.ScenePredicateCalibrationBundle.from_data(fake_complete)
