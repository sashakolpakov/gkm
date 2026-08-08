from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
import bongard.object_bongard_scene_predicate_ir as ir
import bongard.object_scene_visual_frontend as visual_frontend
from bongard.object_bongard_scene_predicate_ir import (
    ObjectBongardScenePredicateIRError,
    SceneAtomKind,
    SceneComparison,
    SceneEntityObservation,
    SceneFormulaNode,
    SceneLanguageSourceMode,
    SceneMergedCell,
    SceneNumericInterval,
    SceneNumericUnit,
    SceneOrientation,
    ScenePanelObservation,
    SceneQuantifier,
    SceneScope,
    SceneSingleObservationPurpose,
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
    ObjectSceneRegisteredTagCell,
    ObjectSceneSoftTag,
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptMode,
    extract_object_scene_proposal_inventory,
    freeze_object_scene_soft_tag_registry,
    observe_object_scene_transcript,
)
from bongard.object_scene_semantic_registry import (
    ObjectSceneSemanticRegistryProposal,
    build_object_scene_semantic_registry_gap,
    prepare_object_scene_semantic_registry_proposal,
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
        1 if bird is Disposition.PRESENT else 2,
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
    mode: str = SceneSingleObservationPurpose.SUPPORT_TRAINING_PASS_A.value,
    empty: bool = False,
) -> ScenePanelObservation:
    sources = tuple(sorted((_raw_digest(panel_id + "-a"), _raw_digest(panel_id + "-b")))) if mode == "repeated_registered_merge" else (
        _raw_digest(panel_id + "-a"),
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
        "panel_registered_tag_cells": (),
        "entities": () if empty else (_entity(bird),),
    }
    provisional = object.__new__(ScenePanelObservation)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ScenePanelObservation(
        **values,
        observation_digest=canonical_digest(ir._observation_content(provisional)),
    )


def _orientation_registry() -> ObjectSceneSoftTagRegistry:
    specs = (
        ("entity", "alpha left-sign object", "group0_positive"),
        ("entity", "beta right-sign object", "group1_positive"),
        ("entity", "gamma neutral object", "bidirectional"),
        ("panel", "delta whole-panel sign", "group0_positive"),
    )
    tags = tuple(
        ObjectSceneSoftTag.create(
            f"tag_{index:04d}",
            scope,
            phrase,
            2,
            (
                {
                    "kind": "shape_appearance",
                    "statement": f"the image visibly has the {phrase}",
                },
            ),
            orientation_constraint=orientation,
        )
        for index, (scope, phrase, orientation) in enumerate(specs)
    )
    values = {
        "source_transcript_digests": (),
        "source_panel_digests": (),
        "tags": tags,
        "dropped_tags": (),
    }
    provisional = object.__new__(ObjectSceneSoftTagRegistry)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ObjectSceneSoftTagRegistry(
        **values,
        registry_digest=canonical_digest(
            visual_frontend._registry_content(provisional)
        ),
    )


def _orientation_panel(
    registry: ObjectSceneSoftTagRegistry, panel_id: str
) -> ScenePanelObservation:
    base = _entity(Disposition.PRESENT)
    entity_tag_ids = tuple(
        item.tag_id for item in registry.tags if item.scope == "entity"
    )
    panel_tag_ids = tuple(
        item.tag_id for item in registry.tags if item.scope == "panel"
    )
    entity = SceneEntityObservation(
        base.object_id,
        base.crop_receipt_digest,
        base.bbox_q16,
        base.area_pixels,
        base.component_count,
        base.emergence_gap_pixels,
        base.overlap_object_ids,
        base.qualitative_cells,
        base.count_cells,
        tuple(
            SceneMergedCell(item, Disposition.PRESENT, None, ())
            for item in entity_tag_ids
        ),
    )
    values = {
        "panel_id": panel_id,
        "panel_digest": _raw_digest(panel_id + "-png"),
        "inventory_digest": _raw_digest(panel_id + "-inventory"),
        "registry_digest": registry.registry_digest,
        "observation_mode": SceneSingleObservationPurpose.SUPPORT_TRAINING_PASS_A.value,
        "source_artifact_digests": (_raw_digest(panel_id + "-artifact"),),
        "source_transcript_digests": (),
        "disposition": Disposition.PRESENT,
        "panel_registered_tag_cells": tuple(
            SceneMergedCell(item, Disposition.PRESENT, None, ())
            for item in panel_tag_ids
        ),
        "entities": (entity,),
    }
    provisional = object.__new__(ScenePanelObservation)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ScenePanelObservation(
        **values,
        observation_digest=canonical_digest(ir._observation_content(provisional)),
    )


@pytest.fixture(scope="module")
def orientation_language():
    registry = _orientation_registry()
    return freeze_object_scene_predicate_language(
        registry,
        (
            _orientation_panel(registry, "orientation_panel_00"),
            _orientation_panel(registry, "orientation_panel_01"),
        ),
        source_mode=SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A,
    )


def _exists_entity_atom(kind: SceneAtomKind, observable_id: str):
    return ir.SceneFormula.quantified(
        SceneScope.ENTITY,
        SceneQuantifier.EXISTS,
        ir.SceneFormula.atom_formula(
            ir.SceneAtom.create(SceneScope.ENTITY, kind, observable_id)
        ),
    )


def _candidate_orientations(language, formula):
    return {
        item.orientation
        for item in enumerate_object_scene_candidates(language)
        if item.formula.formula_digest == formula.formula_digest
    }


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


def _artifact_fixture(*, flip_group0_b: bool = False, b_only_numeric_threshold: bool = False):
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

    def registered_payload(index: int, *, flip: bool, pass_b: bool):
        payload = _payload(inventories[index], registry=registry)
        state = "present" if index == 0 and not flip else "absent"
        for row in payload["objects"]:
            for cell in row["counts"]:
                if cell["observable_id"] == "straight_segment_count" and b_only_numeric_threshold:
                    if index == 0:
                        cell["lower_count"] = 6 if pass_b else 5
                        cell["upper_count"] = 8
                    else:
                        cell["lower_count"] = 0
                        cell["upper_count"] = 4
            for cell in row["observables"]:
                if cell["observable_id"] == "bird_like":
                    cell["state"] = state
                    cell["evidence"] = "bird silhouette visibly supported" if state == "present" else "bird silhouette is not visible"
            for cell in row["registered_tags"]:
                for witness in cell["witness_cells"]:
                    witness["state"] = state
                    witness["evidence"] = (
                        "bird silhouette visibly supported"
                        if state == "present"
                        else "bird silhouette is not visible"
                    )
        return payload

    pass_a = tuple(
        _observe(raw, registered_payload(index, flip=False, pass_b=False), scene_id=f"calibration_panel_{index:02d}", context=f"pass-a-{index}", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
        for index, raw in enumerate(raws)
    )
    pass_b = tuple(
        _observe(raw, registered_payload(index, flip=flip_group0_b and index == 0, pass_b=True), scene_id=f"calibration_panel_{index:02d}", context=f"pass-b-{index}", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
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


def _single_component_candidate(
    language, orientation=SceneOrientation.GROUP0_POSITIVE
):
    return next(
        candidate
        for candidate in enumerate_object_scene_candidates(language)
        if candidate.orientation is orientation
        and candidate.formula.node is SceneFormulaNode.QUANTIFIED
        and candidate.formula.quantifier is SceneQuantifier.EXISTS
        and candidate.formula.children[0].atom is not None
        and candidate.formula.children[0].atom.kind is SceneAtomKind.GEOMETRY
        and candidate.formula.children[0].atom.observable_id == "single_component"
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


def test_registered_macro_repeat_merge_is_witnesswise_before_conjunction():
    tag = ObjectSceneSoftTag.create(
        "tag_0000",
        "entity",
        "opposed mismatched wedge portions",
        2,
        (
            {
                "kind": "part_topology",
                "statement": "two joined wedge-like portions are visibly opposed",
            },
            {
                "kind": "shape_appearance",
                "statement": "the upper and lower portions visibly differ",
            },
        ),
    )
    first = ObjectSceneRegisteredTagCell.create(
        tag,
        [
            {
                "witness_id": "witness_00",
                "state": "absent",
                "evidence": "the joined opposed portions are contradicted",
            },
            {
                "witness_id": "witness_01",
                "state": "present",
                "evidence": "the portions visibly differ",
            },
        ],
    )
    second = ObjectSceneRegisteredTagCell.create(
        tag,
        [
            {
                "witness_id": "witness_00",
                "state": "present",
                "evidence": "the joined opposed portions are visible",
            },
            {
                "witness_id": "witness_01",
                "state": "absent",
                "evidence": "the portions have matching appearance",
            },
        ],
    )
    assert first.disposition is second.disposition is Disposition.CERTIFIED_ABSENT
    merged = ir._merge_registered_macro_cell(first, second, tag=tag)
    assert merged.disposition is Disposition.INDETERMINATE
    assert len(merged.source_cell_digests) == 6

    repeated_first = ir._merge_registered_macro_cell(first, first, tag=tag)
    assert repeated_first.disposition is Disposition.CERTIFIED_ABSENT


def test_positive_closed_language_both_orientations_registry_binding_and_empty_all():
    registry = freeze_object_scene_soft_tag_registry(())
    group0, group1 = _panel("group0", Disposition.PRESENT), _panel("group1", Disposition.CERTIFIED_ABSENT)
    language = freeze_object_scene_predicate_language(
        registry,
        (group0, group1),
        source_mode=SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A,
    )
    candidates = enumerate_object_scene_candidates(language)
    component0 = _single_component_candidate(language)
    assert evaluate_object_scene_candidate(component0, language, group0) is Disposition.PRESENT
    assert evaluate_object_scene_candidate(component0, language, group1) is Disposition.CERTIFIED_ABSENT
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
            if formula.atom is not None:
                assert formula.atom.kind is not SceneAtomKind.QUALITATIVE
            boundary_id = formula.count_boundary_id or (None if formula.atom is None else formula.atom.boundary_id)
            if boundary_id is not None:
                boundary = language.boundary(boundary_id)
                assert boundary.value >= 1
                assert boundary.comparison is not SceneComparison.AT_MOST
    diagnostic = ir.SceneFormula.quantified(
        SceneScope.ENTITY,
        SceneQuantifier.EXISTS,
        ir.SceneFormula.atom_formula(
            ir.SceneAtom.create(
                SceneScope.ENTITY, SceneAtomKind.QUALITATIVE, "bird_like"
            )
        ),
    )
    with pytest.raises(
        ObjectBongardScenePredicateIRError, match="diagnostic-only"
    ):
        ir.ScenePredicateCandidate.create(
            language, SceneOrientation.GROUP0_POSITIVE, diagnostic
        )
    all_single_component = next(item for item in candidates if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.quantifier is SceneQuantifier.ALL and item.formula.children[0].atom is not None and item.formula.children[0].atom.observable_id == "single_component")
    assert evaluate_object_scene_candidate(all_single_component, language, _panel("empty", Disposition.INDETERMINATE, empty=True)) is Disposition.ERROR
    with pytest.raises(ObjectBongardScenePredicateIRError, match="different soft-tag registry"):
        wrong = deepcopy(group0.to_data()); wrong["registry_digest"] = _raw_digest("wrong-registry"); wrong["observation_digest"] = canonical_digest({key: value for key, value in wrong.items() if key != "observation_digest"})
        evaluate_object_scene_candidate(component0, language, ScenePanelObservation.from_data(wrong))
    with pytest.raises(ObjectBongardScenePredicateIRError, match="pass-A support-training"):
        freeze_object_scene_predicate_language(
            registry,
            (_panel("query", Disposition.PRESENT, mode=SceneSingleObservationPurpose.QUERY_EVALUATION.value), group1),
            source_mode=SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A,
        )


def test_tag_orientation_partitions_authorize_only_declared_candidates(
    orientation_language,
):
    language = orientation_language
    assert language.group0_positive_tag_ids == ("tag_0000", "tag_0003")
    assert language.group1_positive_tag_ids == ("tag_0001",)
    assert language.bidirectional_tag_ids == ("tag_0002",)
    assert ir.ScenePredicateLanguage.from_data(language.to_data()) == language
    overlapping = deepcopy(language.to_data())
    overlapping["bidirectional_tag_ids"].append("tag_0000")
    overlapping["bidirectional_tag_ids"].sort()
    overlapping["language_digest"] = canonical_digest(
        {
            key: value
            for key, value in overlapping.items()
            if key != "language_digest"
        }
    )
    with pytest.raises(
        ObjectBongardScenePredicateIRError,
        match="orientation partitions differ",
    ):
        ir.ScenePredicateLanguage.from_data(overlapping)

    group0 = _exists_entity_atom(SceneAtomKind.REGISTERED_TAG, "tag_0000")
    group1 = _exists_entity_atom(SceneAtomKind.REGISTERED_TAG, "tag_0001")
    bidirectional = _exists_entity_atom(
        SceneAtomKind.REGISTERED_TAG, "tag_0002"
    )
    assert _candidate_orientations(language, group0) == {
        SceneOrientation.GROUP0_POSITIVE
    }
    assert _candidate_orientations(language, group1) == {
        SceneOrientation.GROUP1_POSITIVE
    }
    assert _candidate_orientations(language, bidirectional) == set(
        SceneOrientation
    )

    panel_group0 = ir.SceneFormula.atom_formula(
        ir.SceneAtom.create(
            SceneScope.PANEL,
            SceneAtomKind.PANEL_REGISTERED_TAG,
            "tag_0003",
        )
    )
    assert _candidate_orientations(language, panel_group0) == {
        SceneOrientation.GROUP0_POSITIVE
    }


def test_recursive_tag_orientation_intersection_and_tag_free_default(
    orientation_language,
):
    language = orientation_language
    group0_atom = ir.SceneFormula.atom_formula(
        ir.SceneAtom.create(
            SceneScope.ENTITY, SceneAtomKind.REGISTERED_TAG, "tag_0000"
        )
    )
    group1_atom = ir.SceneFormula.atom_formula(
        ir.SceneAtom.create(
            SceneScope.ENTITY, SceneAtomKind.REGISTERED_TAG, "tag_0001"
        )
    )
    geometry_atom = ir.SceneFormula.atom_formula(
        ir.SceneAtom.create(
            SceneScope.ENTITY, SceneAtomKind.GEOMETRY, "single_component"
        )
    )
    tag_and_geometry = ir.SceneFormula.quantified(
        SceneScope.ENTITY,
        SceneQuantifier.EXISTS,
        ir.SceneFormula.conjunction(group0_atom, geometry_atom),
    )
    assert _candidate_orientations(language, tag_and_geometry) == {
        SceneOrientation.GROUP0_POSITIVE
    }

    opposite_tags = ir.SceneFormula.quantified(
        SceneScope.ENTITY,
        SceneQuantifier.EXISTS,
        ir.SceneFormula.conjunction(group0_atom, group1_atom),
    )
    assert ir.authorized_scene_formula_orientations(language, opposite_tags) == ()
    assert opposite_tags.formula_digest in {
        item.formula_digest for item in ir.enumerate_object_scene_formulas(language)
    }
    assert _candidate_orientations(language, opposite_tags) == set()

    geometry_only = _exists_entity_atom(
        SceneAtomKind.GEOMETRY, "single_component"
    )
    assert ir.authorized_scene_formula_orientations(
        language, geometry_only
    ) == tuple(SceneOrientation)
    assert _candidate_orientations(language, geometry_only) == set(
        SceneOrientation
    )


def test_candidate_create_and_decode_reject_forged_opposite_orientation(
    orientation_language,
):
    language = orientation_language
    formula = _exists_entity_atom(SceneAtomKind.REGISTERED_TAG, "tag_0000")
    valid = ir.ScenePredicateCandidate.create(
        language, SceneOrientation.GROUP0_POSITIVE, formula
    )
    serialized = valid.to_data()
    assert "same_language_both_orientations" not in serialized
    assert serialized["formula_authorized_orientations"] == ["group0_positive"]
    assert serialized["post_hoc_orientation_flip"] is False
    assert ir.ScenePredicateCandidate.from_data(
        serialized, language=language
    ) == valid

    with pytest.raises(
        ObjectBongardScenePredicateIRError, match="not authorized"
    ):
        ir.ScenePredicateCandidate.create(
            language, SceneOrientation.GROUP1_POSITIVE, formula
        )

    forged = deepcopy(serialized)
    forged["orientation"] = SceneOrientation.GROUP1_POSITIVE.value
    forged["formula_authorized_orientations"] = [
        SceneOrientation.GROUP1_POSITIVE.value
    ]
    forged["candidate_digest"] = canonical_digest(
        {key: value for key, value in forged.items() if key != "candidate_digest"}
    )
    with pytest.raises(
        ObjectBongardScenePredicateIRError,
        match="serialized orientation authorization differs",
    ):
        ir.ScenePredicateCandidate.from_data(forged, language=language)


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


def test_pass_b_cannot_add_a_numeric_separator_or_refit_the_language():
    registry, discovery, pass_a, pass_b, roles = _artifact_fixture(
        b_only_numeric_threshold=True
    )
    bundle = build_object_bongard_scene_predicate_calibration_bundle(
        registry, discovery, pass_a, pass_b, roles
    )
    language = ir.ScenePredicateLanguage.from_data(bundle.version_space["language"])
    pass_a_observations = tuple(
        ScenePanelObservation.from_data(item)
        for item in bundle.version_space["pass_a_observations"]
    )
    pass_b_observations = tuple(
        ScenePanelObservation.from_data(item)
        for item in bundle.version_space["pass_b_observations"]
    )
    merged_observations = tuple(
        ScenePanelObservation.from_data(item)
        for item in bundle.version_space["support_observations"]
    )

    assert language.source_mode is SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A
    assert language.support_observation_digests == tuple(
        sorted(item.observation_digest for item in pass_a_observations)
    )
    assert bundle.version_space["language_source_mode"] == "support_training_pass_a"
    assert bundle.version_space["language_source_observation_digests"] == list(
        language.support_observation_digests
    )

    # Pass B tightens the positive interval from [5, 8] to [6, 8].  The old
    # merged-source freezer therefore admitted >=6, which cleanly separates
    # the B and merged rows.  It is intentionally absent from the pass-A-only
    # language even though B makes it look attractive.
    b_only = ir.SceneNumericBoundary.create(
        "boundary_99999",
        "straight_segment_count",
        SceneNumericUnit.COUNT,
        SceneComparison.AT_LEAST,
        6,
        tuple(item.observation_digest for item in pass_b_observations),
    )
    assert [
        ir._compare_interval(item.entities[0].count_cells[0].interval, b_only)
        for item in pass_b_observations
    ] == [Disposition.PRESENT, Disposition.CERTIFIED_ABSENT]
    assert [item.entities[0].count_cells[0].interval.lower for item in merged_observations] == [6, 0]
    assert not any(
        item.observable_id == "straight_segment_count" and item.value == 6
        for item in language.boundaries
    )
    assert any(
        formula.atom is not None and formula.atom.kind is SceneAtomKind.COUNT
        for candidate in enumerate_object_scene_candidates(language)
        for formula in (candidate.formula, *candidate.formula.children)
    )
    assert cold_replay_object_bongard_scene_predicate_calibration_bundle(
        bundle, registry
    ) == bundle


def test_zero_proposal_panels_remain_panel_usable_but_never_certify_entity_or_pair_absence():
    raw = _blank_scene(); inventory = extract_object_scene_proposal_inventory(raw)
    assert inventory.objects == ()
    registry = freeze_object_scene_soft_tag_registry(())
    first = _observe(raw, _payload(inventory, registry=registry), scene_id="blank_panel", context="blank-a", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
    second = _observe(raw, _payload(inventory, registry=registry), scene_id="blank_panel", context="blank-b", mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
    merged = adapt_object_scene_registered_pair("blank_panel", first, second)
    single = adapt_object_scene_registered_single("blank_panel", first)
    assert merged.disposition is single.disposition is Disposition.PRESENT
    assert single.observation_mode == SceneSingleObservationPurpose.QUERY_EVALUATION.value
    language = freeze_object_scene_predicate_language(
        registry,
        (_panel("group0z", Disposition.PRESENT), _panel("group1z", Disposition.CERTIFIED_ABSENT)),
        source_mode=SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A,
    )
    assert evaluate_object_scene_candidate(_single_component_candidate(language), language, merged) is Disposition.ERROR
    pair_exists = next(item for item in enumerate_object_scene_candidates(language) if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.scope is SceneScope.PAIR and item.formula.quantifier is SceneQuantifier.EXISTS)
    pair_all = next(item for item in enumerate_object_scene_candidates(language) if item.orientation is SceneOrientation.GROUP0_POSITIVE and item.formula.scope is SceneScope.PAIR and item.formula.quantifier is SceneQuantifier.ALL)
    one = _panel("one_entity", Disposition.PRESENT)
    assert evaluate_object_scene_candidate(pair_exists, language, one) is Disposition.ERROR
    assert evaluate_object_scene_candidate(pair_all, language, one) is Disposition.ERROR


def test_panel_scoped_soft_tag_is_directly_decidable_with_zero_proposals():
    blank = _blank_scene()
    visible = _scene()
    raws = (blank, visible)
    inventories = tuple(extract_object_scene_proposal_inventory(item) for item in raws)
    discovery = tuple(
        _observe(
            raw,
            _payload(
                inventory,
                open_tags=(),
                panel_open_tags=("balanced panel arrangement",),
            ),
            scene_id=f"panel_scope_discovery_{index}",
            context=f"panel-scope-discovery-{index}",
            mode=ObjectSceneTranscriptMode.DISCOVERY,
        )
        for index, (raw, inventory) in enumerate(
            zip(raws, inventories, strict=True)
        )
    )
    registry = freeze_object_scene_soft_tag_registry(
        tuple(item.transcript for item in discovery)
    )
    assert [(item.scope, item.tag) for item in registry.tags] == [
        ("panel", "balanced panel arrangement")
    ]
    registered = _observe(
        blank,
        _payload(inventories[0], registry=registry),
        scene_id="panel_scope_blank",
        context="panel-scope-registered",
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    observation = adapt_object_scene_registered_single(
        "panel_scope_blank",
        registered,
        purpose=SceneSingleObservationPurpose.SUPPORT_TRAINING_PASS_A,
    )
    assert observation.entities == ()
    assert observation.disposition is Disposition.PRESENT
    language = freeze_object_scene_predicate_language(
        registry,
        (observation,),
        source_mode=SceneLanguageSourceMode.SUPPORT_TRAINING_PASS_A,
    )
    candidates = enumerate_object_scene_candidates(language)
    panel_candidate = next(
        item
        for item in candidates
        if item.orientation is SceneOrientation.GROUP0_POSITIVE
        and item.formula.node is SceneFormulaNode.ATOM
        and item.formula.atom is not None
        and item.formula.atom.kind is SceneAtomKind.PANEL_REGISTERED_TAG
    )
    assert (
        evaluate_object_scene_candidate(panel_candidate, language, observation)
        is Disposition.PRESENT
    )
    assert (
        evaluate_object_scene_candidate(_single_component_candidate(language), language, observation)
        is Disposition.ERROR
    )


def test_bundle_round_trip_cold_replay_registry_provenance_capacity_and_stratified_slate(monkeypatch):
    registry, discovery, pass_a, pass_b, roles = _artifact_fixture()
    bundle = build_object_bongard_scene_predicate_calibration_bundle(registry, discovery, pass_a, pass_b, roles)
    assert bundle.registry_derivation_mode == ir.EXACT_OPEN_TAG_FREQUENCY_REGISTRY_DERIVATION_MODE
    assert bundle.registry_derivation_digest == registry.registry_digest
    assert bundle.version_space["registry_derivation_mode"] == bundle.registry_derivation_mode
    assert bundle.version_space["registry_derivation_digest"] == bundle.registry_derivation_digest
    assert len(bundle.candidates) == 376
    assert len(canonical_json(bundle.to_data())) < 16 * 1024 * 1024
    restored = ir.ScenePredicateCalibrationBundle.from_data(bundle.to_data())
    assert cold_replay_object_bongard_scene_predicate_calibration_bundle(restored, registry) == bundle
    assert any("single component" in str(item) for item in bundle.ranker_slate)
    assert all("fixed_qualitative" not in str(item) for item in bundle.ranker_slate)
    if len(bundle.complete_survivor_digests) > 64:
        complexities = {item["complexity"] for item in bundle.ranker_slate}
        survivor_complexities = {item.complexity for item in bundle.candidates if item.candidate_digest in bundle.complete_survivor_digests}
        assert complexities == survivor_complexities or len(complexities) >= 2
    tampered = deepcopy(bundle.to_data()); tampered["ranker_slate"] = []
    with pytest.raises(ObjectBongardScenePredicateIRError):
        ir.ScenePredicateCalibrationBundle.from_data(tampered)
    tampered_derivation = deepcopy(bundle.to_data())
    tampered_derivation["registry_derivation_digest"] = _raw_digest("not-the-registry")
    tampered_derivation["version_space"]["registry_derivation_digest"] = tampered_derivation["registry_derivation_digest"]
    tampered_derivation["bundle_digest"] = canonical_digest({key: value for key, value in tampered_derivation.items() if key != "bundle_digest"})
    with pytest.raises(ObjectBongardScenePredicateIRError, match="exact registry derivation digest"):
        ir.ScenePredicateCalibrationBundle.from_data(tampered_derivation)
    split_derivation = deepcopy(bundle.to_data())
    split_derivation["version_space"]["registry_derivation_mode"] = "role_aware_semantic_proposal"
    split_derivation["bundle_digest"] = canonical_digest({key: value for key, value in split_derivation.items() if key != "bundle_digest"})
    with pytest.raises(ObjectBongardScenePredicateIRError, match="derivation binding"):
        ir.ScenePredicateCalibrationBundle.from_data(split_derivation)
    language = ir.ScenePredicateLanguage.from_data(bundle.version_space["language"])
    forbidden_boundary = next(item for item in language.boundaries if item.observable_id == "matching_entity_count" and item.comparison is SceneComparison.AT_MOST and item.value >= 1)
    body = ir.SceneFormula.atom_formula(
        ir.SceneAtom.create(
            SceneScope.ENTITY, SceneAtomKind.GEOMETRY, "single_component"
        )
    )
    formula = ir.SceneFormula.quantified(SceneScope.ENTITY, SceneQuantifier.COUNT, body, forbidden_boundary.boundary_id)
    values = {"language_digest": language.language_digest, "orientation": SceneOrientation.GROUP0_POSITIVE, "formula_authorized_orientations": tuple(SceneOrientation), "formula": formula, "complexity": formula.complexity}
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


def test_semantic_registry_derivation_routes_only_to_companion_and_replays_by_proposal_digest(monkeypatch):
    registry, discovery, pass_a, pass_b, roles = _artifact_fixture()
    proposal = SimpleNamespace(proposal_digest=_raw_digest("semantic-proposal"))
    semantic_calls = []

    def semantic_verify(candidate, candidate_registry, candidate_discovery, candidate_roles):
        semantic_calls.append((candidate, candidate_registry, candidate_discovery, candidate_roles))
        return candidate_registry

    def forbidden_exact_verify(*args, **kwargs):
        raise AssertionError("semantic derivation must not invoke exact-frequency verification")

    monkeypatch.setattr(ir, "verify_object_scene_soft_tag_registry", forbidden_exact_verify)
    monkeypatch.setattr(ir, "verify_object_scene_semantic_registry_proposal", semantic_verify)
    monkeypatch.setattr(ir, "SCENE_MAX_ENUMERATED_FORMULAS", 1)
    bundle = build_object_bongard_scene_predicate_calibration_bundle(
        registry,
        discovery,
        pass_a,
        pass_b,
        roles,
        semantic_registry_proposal=proposal,
    )
    assert semantic_calls == [(proposal, registry, discovery, roles)]
    assert bundle.registry_derivation_mode == ir.ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    assert bundle.registry_derivation_digest == proposal.proposal_digest
    assert bundle.version_space["registry_derivation_mode"] == bundle.registry_derivation_mode
    assert bundle.version_space["registry_derivation_digest"] == proposal.proposal_digest
    with pytest.raises(ObjectBongardScenePredicateIRError, match="semantic registry proposal differs"):
        cold_replay_object_bongard_scene_predicate_calibration_bundle(
            bundle,
            registry,
            semantic_registry_proposal=proposal,
            discovery_artifacts=discovery,
            role_rows=roles,
        )
    with pytest.raises(ObjectBongardScenePredicateIRError, match="semantic registry proposal differs"):
        cold_replay_object_bongard_scene_predicate_calibration_bundle(
            bundle,
            registry,
            semantic_registry_proposal=SimpleNamespace(
                proposal_digest=_raw_digest("different-semantic-proposal")
            ),
        )


def test_real_semantic_typed_gap_empty_registry_build_round_trip_and_tamper(monkeypatch):
    raws = tuple(_scene(index) for index in range(12))
    inventories = tuple(extract_object_scene_proposal_inventory(raw) for raw in raws)
    discovery = tuple(
        _observe(
            raw,
            _payload(inventory, open_tags=("bird-like object",)),
            scene_id=f"semantic_gap_panel_{index:02d}",
            context=f"semantic-gap-discovery-{index}",
            mode=ObjectSceneTranscriptMode.DISCOVERY,
        )
        for index, (raw, inventory) in enumerate(
            zip(raws, inventories, strict=True)
        )
    )
    roles = tuple(
        {
            "ordinal": index,
            "neutral_panel_digest": _raw_digest(f"semantic-gap-neutral-{index}"),
            "historical_role": 0 if index < 6 else 1,
            "blind_panel_id": f"semantic_gap_panel_{index:02d}",
        }
        for index in range(12)
    )
    prepared = prepare_object_scene_semantic_registry_proposal(discovery, roles)
    proposal, registry = build_object_scene_semantic_registry_gap(
        prepared,
        "insufficient_discovery_evidence",
    )
    assert proposal.status == "typed_proposal_gap"
    assert registry.tags == ()

    def registered_payload(index: int):
        payload = _payload(inventories[index], registry=registry)
        state = "present" if index < 6 else "absent"
        for row in payload["objects"]:
            for cell in row["observables"]:
                if cell["observable_id"] == "bird_like":
                    cell["state"] = state
                    cell["evidence"] = (
                        "bird silhouette visibly supported"
                        if state == "present"
                        else "bird silhouette is not visible"
                    )
        return payload

    pass_a = tuple(
        _observe(
            raw,
            registered_payload(index),
            scene_id=f"semantic_gap_panel_{index:02d}",
            context=f"semantic-gap-pass-a-{index}",
            mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
            registry=registry,
        )
        for index, raw in enumerate(raws)
    )
    pass_b = tuple(
        _observe(
            raw,
            registered_payload(index),
            scene_id=f"semantic_gap_panel_{index:02d}",
            context=f"semantic-gap-pass-b-{index}",
            mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
            registry=registry,
        )
        for index, raw in enumerate(raws)
    )
    monkeypatch.setattr(ir, "SCENE_MAX_ENUMERATED_FORMULAS", 1)
    bundle = build_object_bongard_scene_predicate_calibration_bundle(
        registry,
        discovery,
        pass_a,
        pass_b,
        roles,
        semantic_registry_proposal=proposal,
    )
    restored_proposal = ObjectSceneSemanticRegistryProposal.from_data(
        proposal.to_data()
    )
    restored_bundle = ir.ScenePredicateCalibrationBundle.from_data(bundle.to_data())
    assert bundle.registry_derivation_mode == ir.ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    assert bundle.registry_derivation_digest == proposal.proposal_digest
    assert ir.ScenePredicateLanguage.from_data(
        bundle.version_space["language"]
    ).registered_tag_ids == ()
    assert cold_replay_object_bongard_scene_predicate_calibration_bundle(
        restored_bundle,
        registry,
        semantic_registry_proposal=restored_proposal,
        discovery_artifacts=discovery,
        role_rows=roles,
    ) == bundle

    tampered = deepcopy(bundle.to_data())
    changed_digest = _raw_digest("different-real-semantic-proposal")
    tampered["registry_derivation_digest"] = changed_digest
    tampered["version_space"]["registry_derivation_digest"] = changed_digest
    tampered["bundle_digest"] = canonical_digest(
        {key: value for key, value in tampered.items() if key != "bundle_digest"}
    )
    with pytest.raises(
        ObjectBongardScenePredicateIRError,
        match="semantic registry proposal differs",
    ):
        cold_replay_object_bongard_scene_predicate_calibration_bundle(
            tampered,
            registry,
            semantic_registry_proposal=proposal,
            discovery_artifacts=discovery,
            role_rows=roles,
        )
