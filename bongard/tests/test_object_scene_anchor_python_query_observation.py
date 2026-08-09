from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorWitnessCell,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_cards import ObjectSceneAnchorCardWitness
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorDecisionManifest
from bongard.object_scene_anchor_observer import (
    freeze_object_scene_anchor_observer_vocabulary,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    _manifest_content,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
    ObjectSceneAnchorSelectionCommitment,
    freeze_object_scene_anchor_python_predicate,
)
from bongard.object_scene_anchor_python_query_observation import (
    ObjectSceneAnchorPythonQueryError,
    ObjectSceneAnchorPythonQueryEvaluation,
    ObjectSceneAnchorPythonQueryObservation,
    ObjectSceneAnchorPythonQueryVocabulary,
    build_object_scene_anchor_python_query_observation,
    cold_verify_object_scene_anchor_python_query_evaluation,
    cold_verify_object_scene_anchor_python_query_observation,
    evaluate_object_scene_anchor_python_query_observation,
    freeze_object_scene_anchor_python_query_vocabulary,
)
from bongard.object_scene_anchor_salience import extract_object_scene_anchor_salience
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorAtomCitation,
    ObjectSceneAnchorObjectWitnessMatrix,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPanelWitnessEvaluation,
    ObjectSceneAnchorPredicateAtom,
    ObjectSceneAnchorPredicateLanguage,
    build_object_scene_anchor_panel_witness_evaluation,
    build_object_scene_anchor_support_version_space,
    evaluate_object_scene_anchor_candidate_on_contrast,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _plus() -> np.ndarray:
    mask = np.zeros((43, 43), dtype=np.bool_)
    mask[21, 7:36] = True
    mask[7:36, 21] = True
    return mask


def _decision(object_id: str) -> ObjectSceneAnchorDecisionManifest:
    return ObjectSceneAnchorDecisionManifest.from_salience(
        extract_object_scene_anchor_salience(_plus(), object_id)
    )


def _panel_manifest(
    index: int,
    decisions: tuple[ObjectSceneAnchorDecisionManifest, ...],
) -> ObjectSceneAnchorPanelDecisionManifest:
    values = {
        "panel_digest": _sha(f"query-panel-{index}"),
        "width_pixels": 64,
        "height_pixels": 64,
        "inventory_digest": _sha(f"query-inventory-{index}"),
        "proposal_count": len(decisions),
        "object_ids": tuple(item.object_id for item in decisions),
        "object_decisions": decisions,
    }
    provisional = object.__new__(ObjectSceneAnchorPanelDecisionManifest)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPanelDecisionManifest(
        **values,
        manifest_digest=canonical_digest(_manifest_content(provisional)),
    )


def _catalogs(manifest, spec):
    return tuple(
        build_object_scene_anchor_binding_catalog(
            decision, spec, expected_object_id=object_id
        )
        for object_id, decision in zip(
            manifest.object_ids, manifest.object_decisions, strict=True
        )
    )


def _catalogs_digest(manifest, spec, catalogs) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-card-panel-binding-catalogs.v1",
            "panel_manifest_digest": manifest.manifest_digest,
            "binding_spec_digest": spec.spec_digest,
            "object_ids": list(manifest.object_ids),
            "catalogs": [item.to_data() for item in catalogs],
            "complete_object_inventory_required": True,
        }
    )


def _language(
    target_manifests: dict[str, ObjectSceneAnchorPanelDecisionManifest],
    *,
    witness_count: int = 3,
) -> ObjectSceneAnchorPredicateLanguage:
    spec = ObjectSceneAnchorBindingSpec.entity()
    statements = (
        "the bound form has a rounded upper contour",
        "the bound form carries a centered circular mark",
        "the bound form contains one continuous outer path",
        "the bound form has four evenly spaced oblique arms",
    )
    witnesses = tuple(
        ObjectSceneAnchorCardWitness.create(
            f"witness_{index:02d}",
            "shape_appearance" if index != 1 else "marking_pattern",
            statement,
        )
        for index, statement in enumerate(statements[:witness_count])
    )
    vocabulary = freeze_object_scene_anchor_observer_vocabulary(witnesses)
    citations = []
    for panel_id in sorted(target_manifests):
        manifest = target_manifests[panel_id]
        catalogs = _catalogs(manifest, spec)
        citations.append(
            ObjectSceneAnchorAtomCitation.create(
                panel_id,
                manifest.manifest_digest,
                _catalogs_digest(manifest, spec, catalogs),
                catalogs[0].bindings[0],
            )
        )
    atoms = tuple(
        ObjectSceneAnchorPredicateAtom.create(
            source_card_digest=_sha(f"query-card-{index}"),
            orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
            binding_spec=spec,
            witness_digests=(entry.witness_digest,),
            positive_support_citations=citations,
        )
        for index, entry in enumerate(vocabulary.entries)
    )
    return ObjectSceneAnchorPredicateLanguage.create(
        source_proposal_digest=_sha("query-proposal"),
        vocabulary=vocabulary,
        atoms=atoms,
    )


def _full_panel_evaluation(
    panel_id: str,
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    language: ObjectSceneAnchorPredicateLanguage,
    state,
) -> ObjectSceneAnchorPanelWitnessEvaluation:
    spec = language.atoms[0].binding_spec
    matrices = []
    for catalog in _catalogs(manifest, spec):
        cells = tuple(
            ObjectSceneAnchorWitnessCell.create(
                binding,
                witness,
                state(catalog.object_id, witness.witness_digest),
            )
            for binding in catalog.bindings
            for witness in language.vocabulary.binding_witness_specs
        )
        matrices.append(
            ObjectSceneAnchorObjectWitnessMatrix.create(
                catalog=catalog,
                vocabulary=language.vocabulary,
                cells=cells,
            )
        )
    return build_object_scene_anchor_panel_witness_evaluation(
        panel_id=panel_id,
        panel_manifest=manifest,
        language=language,
        object_matrices=matrices,
    )


def _selected_cells(predicate, manifest, state):
    return tuple(
        ObjectSceneAnchorWitnessCell.create(
            binding,
            entry.binding_witness_spec,
            state(catalog.object_id, entry.witness_digest),
        )
        for catalog in _catalogs(manifest, predicate.binding_spec)
        for binding in catalog.bindings
        for entry in predicate.affirmative_witness_entries
    )


@pytest.fixture(scope="module")
def setup():
    decisions = (_decision("object_0000"), _decision("object_0001"))
    target_ids = tuple(f"target_{index:02d}" for index in range(6))
    contrast_ids = tuple(f"contrast_{index:02d}" for index in range(6))
    support_manifests = {
        panel_id: _panel_manifest(index, decisions)
        for index, panel_id in enumerate((*target_ids, *contrast_ids))
    }
    language = _language(
        {panel_id: support_manifests[panel_id] for panel_id in target_ids}
    )
    targets = tuple(
        _full_panel_evaluation(
            panel_id,
            support_manifests[panel_id],
            language,
            lambda *_: Disposition.PRESENT,
        )
        for panel_id in target_ids
    )
    contrasts = tuple(
        _full_panel_evaluation(
            panel_id,
            support_manifests[panel_id],
            language,
            lambda *_: Disposition.CERTIFIED_ABSENT,
        )
        for panel_id in contrast_ids
    )
    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )
    entry_by_digest = {
        item.witness_digest: item for item in language.vocabulary.entries
    }
    selected = next(
        item
        for item in version.candidates
        if len(item.witness_digests) == 2
        and {
            entry_by_digest[digest].witness_id for digest in item.witness_digests
        }
        == {"witness_00", "witness_02"}
    )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selected.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("query-selector"),
    )
    predicate = freeze_object_scene_anchor_python_predicate(version, selection)
    query_manifest = _panel_manifest(99, decisions)
    return version, language, predicate, query_manifest


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_vocabulary_is_exact_selected_projection_without_support_language(
    setup,
) -> None:
    _, language, predicate, _ = setup
    vocabulary = freeze_object_scene_anchor_python_query_vocabulary(predicate)
    data = vocabulary.to_data()
    encoded = json.dumps(data, sort_keys=True)
    selected = {item.witness_digest for item in predicate.affirmative_witness_entries}
    excluded = next(
        item
        for item in language.vocabulary.entries
        if item.witness_digest not in selected
    )

    assert vocabulary.binding_spec == predicate.binding_spec
    assert vocabulary.entries == predicate.affirmative_witness_entries
    assert len(vocabulary.entries) == 2
    assert excluded.witness_digest not in encoded
    assert excluded.statement not in encoded
    assert data["support_language_payload_present"] is False
    assert ObjectSceneAnchorPythonQueryVocabulary.from_data(data) == vocabulary

    forbidden_checker = "l" + "ean"
    assert not any(forbidden_checker in key.casefold() for key in _all_keys(data))
    assert not any("query" in key.casefold() for key in _all_keys(predicate.to_data()))


def test_four_witness_predicate_and_query_vocabulary_round_trip(setup) -> None:
    _, _, _, query_manifest = setup
    decisions = query_manifest.object_decisions
    target_ids = tuple(f"four_target_{index:02d}" for index in range(6))
    contrast_ids = tuple(f"four_contrast_{index:02d}" for index in range(6))
    manifests = {
        panel_id: _panel_manifest(index + 300, decisions)
        for index, panel_id in enumerate((*target_ids, *contrast_ids))
    }
    language = _language(
        {panel_id: manifests[panel_id] for panel_id in target_ids},
        witness_count=4,
    )
    targets = tuple(
        _full_panel_evaluation(
            panel_id,
            manifests[panel_id],
            language,
            lambda *_: Disposition.PRESENT,
        )
        for panel_id in target_ids
    )
    contrasts = tuple(
        _full_panel_evaluation(
            panel_id,
            manifests[panel_id],
            language,
            lambda *_: Disposition.CERTIFIED_ABSENT,
        )
        for panel_id in contrast_ids
    )
    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )
    selected = next(
        item for item in version.candidates if len(item.witness_digests) == 4
    )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selected.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("four-query-selector"),
    )
    predicate = freeze_object_scene_anchor_python_predicate(version, selection)
    vocabulary = freeze_object_scene_anchor_python_query_vocabulary(predicate)

    assert len(predicate.affirmative_witness_entries) == 4
    assert len(vocabulary.entries) == 4
    assert ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data()) == predicate
    assert ObjectSceneAnchorPythonQueryVocabulary.from_data(
        vocabulary.to_data()
    ) == vocabulary


def test_build_requires_exact_complete_selected_binding_rectangle(setup) -> None:
    _, language, predicate, manifest = setup
    cells = _selected_cells(
        predicate, manifest, lambda *_: Disposition.PRESENT
    )
    observation = build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id="neutral_00",
        panel_manifest=manifest,
        cells=cells,
    )

    assert observation.object_ids == ("object_0000", "object_0001")
    assert len(observation.objects) == 2
    assert all(len(item.rows) == 1 for item in observation.objects)
    assert all(len(item.rows[0].cells) == 2 for item in observation.objects)
    assert build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id="neutral_00",
        panel_manifest=manifest,
        cells=tuple(reversed(cells)),
    ) == observation

    with pytest.raises(ObjectSceneAnchorPythonQueryError, match="exactly cover"):
        build_object_scene_anchor_python_query_observation(
            predicate=predicate,
            panel_id="neutral_00",
            panel_manifest=manifest,
            cells=cells[:-1],
        )
    with pytest.raises(ObjectSceneAnchorPythonQueryError, match="duplicate"):
        build_object_scene_anchor_python_query_observation(
            predicate=predicate,
            panel_id="neutral_00",
            panel_manifest=manifest,
            cells=(*cells, cells[0]),
        )

    selected = {item.witness_digest for item in predicate.affirmative_witness_entries}
    excluded = next(
        item
        for item in language.vocabulary.entries
        if item.witness_digest not in selected
    )
    binding = _catalogs(manifest, predicate.binding_spec)[0].bindings[0]
    foreign = ObjectSceneAnchorWitnessCell.create(
        binding, excluded.binding_witness_spec, Disposition.PRESENT
    )
    with pytest.raises(ObjectSceneAnchorPythonQueryError, match="selected vocabulary"):
        build_object_scene_anchor_python_query_observation(
            predicate=predicate,
            panel_id="neutral_00",
            panel_manifest=manifest,
            cells=(*cells, foreign),
        )


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("present", Disposition.PRESENT),
        ("crossed", Disposition.CERTIFIED_ABSENT),
        ("uncertain", Disposition.INDETERMINATE),
        ("failed", Disposition.ERROR),
    ),
)
def test_query_evaluation_matches_exhaustive_same_binding_semantics(
    setup, case, expected
) -> None:
    _, language, predicate, manifest = setup
    first, second = predicate.candidate.witness_digests

    def state(object_id, witness_digest):
        if witness_digest not in (first, second):
            return Disposition.CERTIFIED_ABSENT
        if case == "present":
            return Disposition.PRESENT
        if case == "crossed":
            if object_id == "object_0000":
                return (
                    Disposition.PRESENT
                    if witness_digest == first
                    else Disposition.CERTIFIED_ABSENT
                )
            return (
                Disposition.PRESENT
                if witness_digest == second
                else Disposition.CERTIFIED_ABSENT
            )
        if case == "uncertain":
            if object_id == "object_0000":
                return (
                    Disposition.INDETERMINATE
                    if witness_digest == first
                    else Disposition.PRESENT
                )
            return Disposition.CERTIFIED_ABSENT
        if object_id == "object_0000":
            return Disposition.PRESENT
        return (
            Disposition.ERROR
            if witness_digest == first
            else Disposition.CERTIFIED_ABSENT
        )

    observation = build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id="neutral_00",
        panel_manifest=manifest,
        cells=_selected_cells(predicate, manifest, state),
    )
    evaluation = evaluate_object_scene_anchor_python_query_observation(
        predicate, observation
    )
    full_panel = _full_panel_evaluation(
        "neutral_00", manifest, language, state
    )

    assert evaluation.disposition is expected
    assert evaluation.disposition is evaluate_object_scene_anchor_candidate_on_contrast(
        predicate.candidate, language, full_panel
    )


def test_strict_round_trip_and_model_free_cold_replay(setup) -> None:
    version, _, predicate, manifest = setup
    cells = _selected_cells(
        predicate, manifest, lambda *_: Disposition.INDETERMINATE
    )
    observation = build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id="neutral_00",
        panel_manifest=manifest,
        cells=cells,
    )
    evaluation = evaluate_object_scene_anchor_python_query_observation(
        predicate, observation
    )

    assert ObjectSceneAnchorPythonQueryObservation.from_data(
        observation.to_data()
    ) == observation
    assert cold_verify_object_scene_anchor_python_query_observation(
        observation,
        predicate=predicate,
        panel_manifest=manifest,
    ) == observation
    assert ObjectSceneAnchorPythonQueryEvaluation.from_data(
        evaluation.to_data()
    ) == evaluation
    assert cold_verify_object_scene_anchor_python_query_evaluation(
        evaluation,
        predicate=predicate,
        observation=observation,
    ) == evaluation

    extra = deepcopy(observation.to_data())
    extra["full_language"] = {}
    with pytest.raises(ObjectSceneAnchorPythonQueryError, match="fields differ"):
        ObjectSceneAnchorPythonQueryObservation.from_data(extra)

    changed = deepcopy(observation.to_data())
    changed["objects"][0]["rows"][0]["cells"][0]["disposition"] = "present"
    with pytest.raises(ValueError, match="digest"):
        ObjectSceneAnchorPythonQueryObservation.from_data(changed)

    other_candidate = next(
        item for item in version.candidates if len(item.witness_digests) == 1
    )
    other_selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=other_candidate.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("other-query-selector"),
    )
    other_predicate = freeze_object_scene_anchor_python_predicate(
        version, other_selection
    )
    with pytest.raises(ObjectSceneAnchorPythonQueryError, match="frozen predicate"):
        evaluate_object_scene_anchor_python_query_observation(
            other_predicate, observation
        )


def test_empty_object_inventory_is_certified_absence(setup) -> None:
    _, _, predicate, _ = setup
    empty_manifest = _panel_manifest(101, ())
    observation = build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id="neutral_empty",
        panel_manifest=empty_manifest,
        cells=(),
    )

    assert observation.objects == ()
    assert evaluate_object_scene_anchor_python_query_observation(
        predicate, observation
    ).disposition is Disposition.CERTIFIED_ABSENT
