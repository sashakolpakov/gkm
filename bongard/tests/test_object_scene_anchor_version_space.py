from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorWitnessCell,
    ObjectSceneResolvedAnchorBinding,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_cards import (
    ObjectSceneAnchorCardWitness,
    build_object_scene_anchor_card_proposal,
)
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorDecisionManifest
from bongard.object_scene_anchor_observer import (
    freeze_object_scene_anchor_observer_vocabulary,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    _manifest_content,
)
from bongard.object_scene_anchor_salience import extract_object_scene_anchor_salience
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorAtomCitation,
    ObjectSceneAnchorObjectWitnessMatrix,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPanelWitnessEvaluation,
    ObjectSceneAnchorPredicateAtom,
    ObjectSceneAnchorPredicateLanguage,
    ObjectSceneAnchorSupportGapKind,
    ObjectSceneAnchorSupportVersionSpace,
    ObjectSceneAnchorVersionSpaceError,
    build_object_scene_anchor_panel_witness_evaluation,
    build_object_scene_anchor_support_version_space,
    cold_verify_object_scene_anchor_support_version_space,
    enumerate_object_scene_anchor_candidates,
    evaluate_object_scene_anchor_candidate_on_contrast,
    evaluate_object_scene_anchor_candidate_on_target,
    project_object_scene_anchor_card_proposal,
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
        "panel_digest": _sha(f"panel-{index}"),
        "width_pixels": 64,
        "height_pixels": 64,
        "inventory_digest": _sha(f"inventory-{index}"),
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


def _catalogs(
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    spec: ObjectSceneAnchorBindingSpec,
):
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
    manifests: dict[str, ObjectSceneAnchorPanelDecisionManifest],
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
            "shape_appearance" if index % 2 == 0 else "part_topology",
            statements[index],
        )
        for index in range(witness_count)
    )
    vocabulary = freeze_object_scene_anchor_observer_vocabulary(witnesses)
    citations = []
    for panel_id in sorted(manifests):
        manifest = manifests[panel_id]
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
            source_card_digest=_sha(f"card-{index}"),
            orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
            binding_spec=spec,
            witness_digests=(witness.witness_digest,),
            positive_support_citations=citations,
        )
        for index, witness in enumerate(witnesses)
    )
    return ObjectSceneAnchorPredicateLanguage.create(
        source_proposal_digest=_sha("proposal"),
        vocabulary=vocabulary,
        atoms=atoms,
    )


def _panel_evaluation(
    panel_id: str,
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    language: ObjectSceneAnchorPredicateLanguage,
    state,
) -> ObjectSceneAnchorPanelWitnessEvaluation:
    specs = {
        item.binding_spec.spec_digest: item.binding_spec for item in language.atoms
    }
    matrices = []
    for spec in tuple(specs[key] for key in sorted(specs)):
        for catalog in _catalogs(manifest, spec):
            cells = tuple(
                ObjectSceneAnchorWitnessCell.create(
                    binding,
                    witness,
                    state(panel_id, catalog.object_id, witness.witness_digest),
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


@pytest.fixture(scope="module")
def setup():
    decisions = (_decision("object_0000"), _decision("object_0001"))
    target_ids = tuple(f"target_{index:02d}" for index in range(6))
    contrast_ids = tuple(f"contrast_{index:02d}" for index in range(6))
    manifests = {
        panel_id: _panel_manifest(index, decisions)
        for index, panel_id in enumerate((*target_ids, *contrast_ids))
    }
    language = _language({panel_id: manifests[panel_id] for panel_id in target_ids})
    witness_digests = tuple(item.witness_digest for item in language.vocabulary.entries)
    return manifests, target_ids, contrast_ids, language, witness_digests


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_complete_witness_atom_and_conjunction_inventory(setup) -> None:
    _, _, _, language, _ = setup
    candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )

    assert len(language.atoms) == 3
    assert all(len(item.witness_digests) == 1 for item in language.atoms)
    assert [len(item.atom_digests) for item in candidates] == [1, 1, 1, 2, 2, 2, 3]
    assert all(
        tuple(item.atom_digests) == tuple(sorted(item.atom_digests))
        for item in candidates
    )
    keys = tuple(_all_keys(language.to_data()))
    assert not any("entry_digest" in key for key in keys)
    assert not any("salience_artifact" in key for key in keys)
    assert not any("lean" in key.casefold() for key in keys)


def test_four_witness_card_conjunction_is_not_silently_omitted(setup) -> None:
    manifests, target_ids, _, _, _ = setup
    language = _language(
        {panel_id: manifests[panel_id] for panel_id in target_ids},
        witness_count=4,
    )
    candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )

    assert len(candidates) == 15
    four = tuple(item for item in candidates if len(item.witness_digests) == 4)
    assert len(four) == 1
    assert four[0].witness_digests == tuple(
        sorted(item.witness_digest for item in language.vocabulary.entries)
    )


def test_target_uses_exact_citation_while_contrast_exhausts_all_objects(setup) -> None:
    manifests, target_ids, contrast_ids, language, witness_digests = setup
    first_witness = witness_digests[0]

    def target_state(panel_id, object_id, witness_digest):
        # An unrelated object error must not poison an exact cited target.
        return Disposition.PRESENT if object_id == "object_0000" else Disposition.ERROR

    def contrast_state(panel_id, object_id, witness_digest):
        if (
            panel_id == contrast_ids[0]
            and object_id == "object_0001"
            and witness_digest == first_witness
        ):
            return Disposition.PRESENT
        return Disposition.CERTIFIED_ABSENT

    targets = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, target_state)
        for panel_id in target_ids
    )
    contrasts = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, contrast_state)
        for panel_id in contrast_ids
    )
    candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    atom = next(item for item in candidates if item.witness_digests == (first_witness,))

    assert evaluate_object_scene_anchor_candidate_on_target(
        atom, language, targets[0]
    ) is Disposition.PRESENT
    assert evaluate_object_scene_anchor_candidate_on_contrast(
        atom, language, contrasts[0]
    ) is Disposition.PRESENT

    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )
    survivors = tuple(
        item
        for item in version.candidates
        if item.candidate_digest in version.survivor_candidate_digests
    )
    assert survivors
    assert atom.candidate_digest not in version.survivor_candidate_digests
    assert ObjectSceneAnchorSupportVersionSpace.from_data(version.to_data()) == version
    assert cold_verify_object_scene_anchor_support_version_space(
        version,
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    ) == version


def test_conjunction_cannot_pool_different_bindings_and_error_blocks_absence(setup) -> None:
    manifests, _, contrast_ids, language, witness_digests = setup
    first, second = witness_digests[:2]

    def crossed(panel_id, object_id, witness_digest):
        if object_id == "object_0000":
            return Disposition.PRESENT if witness_digest == first else Disposition.CERTIFIED_ABSENT
        return Disposition.PRESENT if witness_digest == second else Disposition.CERTIFIED_ABSENT

    panel = _panel_evaluation(
        contrast_ids[0], manifests[contrast_ids[0]], language, crossed
    )
    candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    conjunction = next(
        item for item in candidates if item.witness_digests == tuple(sorted((first, second)))
    )
    assert evaluate_object_scene_anchor_candidate_on_contrast(
        conjunction, language, panel
    ) is Disposition.CERTIFIED_ABSENT

    def failed(panel_id, object_id, witness_digest):
        if object_id == "object_0001" and witness_digest == first:
            return Disposition.ERROR
        return Disposition.CERTIFIED_ABSENT

    failed_panel = _panel_evaluation(
        contrast_ids[0], manifests[contrast_ids[0]], language, failed
    )
    atom = next(item for item in candidates if item.witness_digests == (first,))
    assert evaluate_object_scene_anchor_candidate_on_contrast(
        atom, language, failed_panel
    ) is Disposition.ERROR


@pytest.mark.parametrize(
    ("uncertain", "expected_kind"),
    (
        (False, ObjectSceneAnchorSupportGapKind.LANGUAGE_GAP),
        (True, ObjectSceneAnchorSupportGapKind.WITNESS_GAP),
    ),
)
def test_empty_space_distinguishes_language_and_witness_gaps(
    setup, uncertain, expected_kind
) -> None:
    manifests, target_ids, contrast_ids, _, _ = setup
    language = _language({panel_id: manifests[panel_id] for panel_id in target_ids}, witness_count=1)

    def targets_state(panel_id, object_id, witness_digest):
        return Disposition.PRESENT

    def contrast_state(panel_id, object_id, witness_digest):
        if panel_id == contrast_ids[0] and object_id == "object_0000":
            return Disposition.INDETERMINATE if uncertain else Disposition.PRESENT
        return Disposition.CERTIFIED_ABSENT

    targets = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, targets_state)
        for panel_id in target_ids
    )
    contrasts = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, contrast_state)
        for panel_id in contrast_ids
    )
    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )

    assert version.survivor_candidate_digests == ()
    assert version.gap is not None
    assert version.gap.kind is expected_kind


def test_panel_matrix_rejects_omission_reordering_and_tamper(setup) -> None:
    manifests, target_ids, _, language, _ = setup
    panel_id = target_ids[0]
    manifest = manifests[panel_id]
    spec = language.atoms[0].binding_spec
    matrices = []
    for catalog in _catalogs(manifest, spec):
        cells = tuple(
            ObjectSceneAnchorWitnessCell.create(
                binding, witness, Disposition.PRESENT
            )
            for binding in catalog.bindings
            for witness in language.vocabulary.binding_witness_specs
        )
        matrices.append(
            ObjectSceneAnchorObjectWitnessMatrix.create(
                catalog=catalog, vocabulary=language.vocabulary, cells=cells
            )
        )

    with pytest.raises(ObjectSceneAnchorVersionSpaceError, match="omit, reorder"):
        build_object_scene_anchor_panel_witness_evaluation(
            panel_id=panel_id,
            panel_manifest=manifest,
            language=language,
            object_matrices=tuple(reversed(matrices)),
        )
    with pytest.raises(ObjectSceneAnchorVersionSpaceError, match="complete"):
        ObjectSceneAnchorObjectWitnessMatrix.create(
            catalog=matrices[0].catalog,
            vocabulary=language.vocabulary,
            cells=matrices[0].rows[0].cells[:-1],
        )

    panel = build_object_scene_anchor_panel_witness_evaluation(
        panel_id=panel_id,
        panel_manifest=manifest,
        language=language,
        object_matrices=matrices,
    )
    assert ObjectSceneAnchorPanelWitnessEvaluation.from_data(panel.to_data()) == panel
    tampered = deepcopy(panel.to_data())
    tampered["evaluation_digest"] = "0" * 64
    with pytest.raises(ObjectSceneAnchorVersionSpaceError, match="digest"):
        ObjectSceneAnchorPanelWitnessEvaluation.from_data(tampered)


def test_target_structural_mismatch_is_error_never_negative(setup) -> None:
    manifests, target_ids, _, base_language, _ = setup
    panel_id = target_ids[0]
    base_atom = base_language.atoms[0]
    base_citation = next(
        item for item in base_atom.positive_support_citations if item.panel_id == panel_id
    )

    def replace_citation(replacement):
        citations = tuple(
            replacement if item.panel_id == panel_id else item
            for item in base_atom.positive_support_citations
        )
        replacement_atom = ObjectSceneAnchorPredicateAtom.create(
            source_card_digest=base_atom.source_card_digest,
            orientation=base_atom.orientation,
            binding_spec=base_atom.binding_spec,
            witness_digests=base_atom.witness_digests,
            positive_support_citations=citations,
        )
        return ObjectSceneAnchorPredicateLanguage.create(
            source_proposal_digest=_sha(f"replacement-{replacement.citation_digest}"),
            vocabulary=base_language.vocabulary,
            atoms=(replacement_atom, *base_language.atoms[1:]),
        )

    wrong_manifest = ObjectSceneAnchorAtomCitation.create(
        panel_id,
        _sha("wrong-panel-manifest"),
        base_citation.binding_catalogs_digest,
        base_citation.binding,
    )
    missing_binding = ObjectSceneResolvedAnchorBinding.create(
        binding_id=base_citation.binding.binding_id,
        object_id=base_citation.binding.object_id,
        decision_manifest_digest=base_citation.binding.decision_manifest_digest,
        spec_digest=base_citation.binding.spec_digest,
        anchor_kind=base_citation.binding.anchor_kind,
        anchor_id=base_citation.binding.anchor_id,
        anchor_digest=_sha("missing-anchor"),
        selected_graph_digest=base_citation.binding.selected_graph_digest,
    )
    missing_citation = ObjectSceneAnchorAtomCitation.create(
        panel_id,
        base_citation.panel_manifest_digest,
        base_citation.binding_catalogs_digest,
        missing_binding,
    )

    for replacement in (wrong_manifest, missing_citation):
        language = replace_citation(replacement)
        panel = _panel_evaluation(
            panel_id,
            manifests[panel_id],
            language,
            lambda *_: Disposition.PRESENT,
        )
        projected_atom = next(
            item
            for item in language.atoms
            if item.source_card_digest == base_atom.source_card_digest
        )
        candidate = next(
            item
            for item in enumerate_object_scene_anchor_candidates(
                language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
            )
            if item.atom_digests == (projected_atom.atom_digest,)
        )
        assert evaluate_object_scene_anchor_candidate_on_target(
            candidate, language, panel
        ) is Disposition.ERROR


def test_target_cross_atom_citation_mismatch_is_error_never_absence(setup) -> None:
    manifests, target_ids, _, base_language, _ = setup
    panel_id = target_ids[0]
    manifest = manifests[panel_id]
    first_atom, second_atom, *remaining_atoms = base_language.atoms
    catalogs = _catalogs(manifest, second_atom.binding_spec)
    replacement_citation = ObjectSceneAnchorAtomCitation.create(
        panel_id,
        manifest.manifest_digest,
        _catalogs_digest(manifest, second_atom.binding_spec, catalogs),
        catalogs[1].bindings[0],
    )
    replacement_atom = ObjectSceneAnchorPredicateAtom.create(
        source_card_digest=second_atom.source_card_digest,
        orientation=second_atom.orientation,
        binding_spec=second_atom.binding_spec,
        witness_digests=second_atom.witness_digests,
        positive_support_citations=tuple(
            replacement_citation if item.panel_id == panel_id else item
            for item in second_atom.positive_support_citations
        ),
    )
    language = ObjectSceneAnchorPredicateLanguage.create(
        source_proposal_digest=_sha("cross-atom-citation-mismatch"),
        vocabulary=base_language.vocabulary,
        atoms=(first_atom, replacement_atom, *remaining_atoms),
    )
    panel = _panel_evaluation(
        panel_id,
        manifest,
        language,
        lambda *_: Disposition.PRESENT,
    )
    candidate = next(
        item
        for item in enumerate_object_scene_anchor_candidates(
            language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
        )
        if item.atom_digests
        == tuple(sorted((first_atom.atom_digest, replacement_atom.atom_digest)))
    )

    assert evaluate_object_scene_anchor_candidate_on_target(
        candidate, language, panel
    ) is Disposition.ERROR


def test_exact_card_adapter_emits_one_atom_per_affirmative_witness(setup) -> None:
    manifests, target_ids, contrast_ids, _, _ = setup
    side0 = {
        f"panel_{index:03d}": manifests[item]
        for index, item in enumerate(target_ids)
    }
    side1 = {
        f"panel_{index + 6:03d}": manifests[item]
        for index, item in enumerate(contrast_ids)
    }

    def card(aliases, phrase, witnesses):
        return {
            "phrase": phrase,
            "binding_spec": ObjectSceneAnchorBindingSpec.entity().to_data(),
            "required_witnesses": witnesses,
            "accepted_variants": [],
            "near_miss_boundaries": [],
            "positive_support_citations": [
                {
                    "panel_alias": alias,
                    "object_id": "object_0000",
                    "anchor_id": "entity",
                }
                for alias in sorted(aliases)
            ],
        }

    payload = {
        "side0_positive": [
            card(
                tuple(side0),
                "rounded marked form",
                [
                    {
                        "kind": "shape_appearance",
                        "statement": "the bound form has a rounded upper contour",
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": "the bound form carries a centered circular mark",
                    },
                ],
            )
        ],
        "side1_positive": [
            card(
                tuple(side1),
                "angular marked form",
                [
                    {
                        "kind": "shape_appearance",
                        "statement": "the bound form has an angular upper contour",
                    }
                ],
            )
        ],
    }
    proposal = build_object_scene_anchor_card_proposal(
        payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    language = project_object_scene_anchor_card_proposal(proposal)

    assert len(language.atoms) == 3
    assert all(len(item.witness_digests) == 1 for item in language.atoms)
    side0_candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    assert [len(item.atom_digests) for item in side0_candidates] == [1, 1, 2]
