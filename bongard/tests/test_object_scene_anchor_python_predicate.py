from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

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
import bongard.object_scene_anchor_python_predicate as predicate_module
from bongard.object_scene_anchor_python_predicate import (
    OBJECT_SCENE_ANCHOR_CLOSED_FORMULA,
    ObjectSceneAnchorPythonPredicate,
    ObjectSceneAnchorPythonPredicateError,
    ObjectSceneAnchorPythonPredicateEvaluation,
    ObjectSceneAnchorSelectionCommitment,
    cold_verify_object_scene_anchor_python_predicate,
    cold_verify_object_scene_anchor_python_predicate_evaluation,
    evaluate_object_scene_anchor_python_predicate,
    freeze_object_scene_anchor_python_predicate,
    object_scene_anchor_selection_commitment_from_rank_response,
)
from bongard.object_scene_anchor_salience import extract_object_scene_anchor_salience
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorAtomCitation,
    ObjectSceneAnchorObjectWitnessMatrix,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPanelWitnessEvaluation,
    ObjectSceneAnchorPredicateAtom,
    ObjectSceneAnchorPredicateLanguage,
    ObjectSceneAnchorSupportVersionSpace,
    build_object_scene_anchor_panel_witness_evaluation,
    build_object_scene_anchor_support_version_space,
    enumerate_object_scene_anchor_candidates,
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
) -> ObjectSceneAnchorPredicateLanguage:
    spec = ObjectSceneAnchorBindingSpec.entity()
    witnesses = (
        ObjectSceneAnchorCardWitness.create(
            "witness_00",
            "shape_appearance",
            "the bound form has a rounded upper contour",
        ),
        ObjectSceneAnchorCardWitness.create(
            "witness_01",
            "marking_pattern",
            "the bound form carries a centered circular mark",
        ),
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
    spec = language.atoms[0].binding_spec
    matrices = []
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


def _version_fixture(contrast_state):
    decisions = (_decision("object_0000"), _decision("object_0001"))
    target_ids = tuple(f"target_{index:02d}" for index in range(6))
    contrast_ids = tuple(f"contrast_{index:02d}" for index in range(6))
    manifests = {
        panel_id: _panel_manifest(index, decisions)
        for index, panel_id in enumerate((*target_ids, *contrast_ids))
    }
    language = _language(
        {panel_id: manifests[panel_id] for panel_id in target_ids}
    )
    targets = tuple(
        _panel_evaluation(
            panel_id,
            manifests[panel_id],
            language,
            lambda *_: Disposition.PRESENT,
        )
        for panel_id in target_ids
    )
    contrasts = tuple(
        _panel_evaluation(
            panel_id, manifests[panel_id], language, contrast_state
        )
        for panel_id in contrast_ids
    )
    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )
    return version, language, manifests


@pytest.fixture(scope="module")
def frozen_setup():
    version, language, manifests = _version_fixture(
        lambda *_: Disposition.CERTIFIED_ABSENT
    )
    selected = next(
        item for item in version.candidates if len(item.witness_digests) == 2
    )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selected.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("test-selector-record"),
    )
    predicate = freeze_object_scene_anchor_python_predicate(version, selection)
    return version, language, manifests, selection, predicate


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_freeze_persists_exact_positive_formula_and_commitments(frozen_setup) -> None:
    version, language, _, selection, predicate = frozen_setup

    assert predicate.candidate.candidate_digest == selection.selected_candidate_digest
    assert predicate.version_space_digest == version.version_space_digest
    assert predicate.language_digest == language.language_digest
    assert predicate.binding_spec.spec_digest == predicate.candidate.binding_spec_digest
    assert tuple(
        item.witness_digest for item in predicate.affirmative_witness_entries
    ) == predicate.candidate.witness_digests
    assert tuple(
        item.statement for item in predicate.affirmative_witness_entries
    ) == tuple(
        next(
            entry.statement
            for entry in language.vocabulary.entries
            if entry.witness_digest == digest
        )
        for digest in predicate.candidate.witness_digests
    )
    assert predicate.formula.to_data()["closed_formula"] == (
        OBJECT_SCENE_ANCHOR_CLOSED_FORMULA
    )
    assert ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data()) == predicate
    assert cold_verify_object_scene_anchor_python_predicate(
        predicate,
        version_space=version,
        selection_commitment=selection,
    ) == predicate

    keys = tuple(_all_keys(predicate.to_data()))
    forbidden_checker = "l" + "ean"
    assert not any(forbidden_checker in key.casefold() for key in keys)
    assert not any("query" in key.casefold() for key in keys)
    assert not any("citation" in key.casefold() for key in keys)


def test_selection_rejects_non_survivor_and_empty_space() -> None:
    language_digest_holder: list[str] = []

    def one_witness_leaks(panel_id, object_id, witness_digest):
        if witness_digest not in language_digest_holder:
            language_digest_holder.append(witness_digest)
        return (
            Disposition.PRESENT
            if panel_id == "contrast_00"
            and object_id == "object_0000"
            and witness_digest == language_digest_holder[0]
            else Disposition.CERTIFIED_ABSENT
        )

    version, language, _ = _version_fixture(
        one_witness_leaks
    )

    candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    rejected = next(
        item
        for item in candidates
        if item.candidate_digest not in version.survivor_candidate_digests
    )
    with pytest.raises(ObjectSceneAnchorPythonPredicateError, match="not an exact"):
        ObjectSceneAnchorSelectionCommitment.create(
            version,
            selected_candidate_digest=rejected.candidate_digest,
            selection_kind="external_exact_selection",
            selector_record_digest=_sha("selector"),
        )

    empty, _, _ = _version_fixture(lambda *_: Disposition.PRESENT)
    assert empty.survivor_candidate_digests == ()
    with pytest.raises(ObjectSceneAnchorPythonPredicateError, match="nonempty"):
        ObjectSceneAnchorSelectionCommitment.create(
            empty,
            selected_candidate_digest=empty.candidates[0].candidate_digest,
            selection_kind="external_exact_selection",
            selector_record_digest=_sha("selector"),
        )


def test_strict_decode_and_cold_replay_reject_tamper(frozen_setup) -> None:
    version, _, _, selection, predicate = frozen_setup
    extra = deepcopy(predicate.to_data())
    extra["extra"] = True
    with pytest.raises(ObjectSceneAnchorPythonPredicateError, match="fields differ"):
        ObjectSceneAnchorPythonPredicate.from_data(extra)

    changed = deepcopy(predicate.to_data())
    changed["affirmative_witness_entries"][0]["statement"] = (
        "the bound form has a sharply pointed upper contour"
    )
    with pytest.raises(
        (ObjectSceneAnchorPythonPredicateError, ValueError), match="digest"
    ):
        ObjectSceneAnchorPythonPredicate.from_data(changed)

    other_selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selection.selected_candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("different-selector-record"),
    )
    with pytest.raises(ObjectSceneAnchorPythonPredicateError, match="cold replay"):
        cold_verify_object_scene_anchor_python_predicate(
            predicate,
            version_space=version,
            selection_commitment=other_selection,
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
def test_neutral_evaluation_is_exhaustive_same_binding_four_way(
    frozen_setup, case, expected
) -> None:
    _, language, _, _, predicate = frozen_setup
    manifest = _panel_manifest(
        99, (_decision("object_0000"), _decision("object_0001"))
    )
    first, second = predicate.candidate.witness_digests

    def state(panel_id, object_id, witness_digest):
        del panel_id
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

    panel = _panel_evaluation("neutral_00", manifest, language, state)
    evaluation = evaluate_object_scene_anchor_python_predicate(predicate, panel)

    assert evaluation.disposition is expected
    assert evaluation.disposition is evaluate_object_scene_anchor_candidate_on_contrast(
        predicate.candidate, language, panel
    )
    assert ObjectSceneAnchorPythonPredicateEvaluation.from_data(
        evaluation.to_data()
    ) == evaluation
    assert cold_verify_object_scene_anchor_python_predicate_evaluation(
        evaluation, predicate=predicate, panel=panel
    ) == evaluation


def test_rank_response_adapter_is_lazy_and_exact(frozen_setup, monkeypatch) -> None:
    version, _, _, _, _ = frozen_setup
    selected = version.survivor_candidate_digests[-1]

    class FakeRankInput:
        version_space_digest = version.version_space_digest
        version_space_algorithm_digest = version.algorithm_digest
        language_digest = version.language.language_digest
        survivor_candidate_digests = version.survivor_candidate_digests

    class FakeRankResponse:
        _instances = {}

        def __init__(self):
            self.rank_input = FakeRankInput()
            self.version_space_digest = version.version_space_digest
            self.selected_candidate_digest = selected
            self.response_digest = _sha("rank-response")
            self._instances[self.response_digest] = self

        def to_data(self):
            return {"response_digest": self.response_digest}

        @classmethod
        def from_data(cls, value):
            return cls._instances[value["response_digest"]]

    fake_module = SimpleNamespace(ObjectSceneAnchorRankResponse=FakeRankResponse)
    imported = []

    def fake_import(name):
        imported.append(name)
        return fake_module

    monkeypatch.setattr(predicate_module.importlib, "import_module", fake_import)
    response = FakeRankResponse()
    commitment = object_scene_anchor_selection_commitment_from_rank_response(
        response, version
    )

    assert imported == ["bongard.object_scene_anchor_candidate_ranker"]
    assert commitment.selected_candidate_digest == selected
    assert commitment.selector_record_digest == response.response_digest
    assert commitment.selection_kind == "exact_rank_response"

    with pytest.raises(TypeError, match="exact anchor rank-response"):
        object_scene_anchor_selection_commitment_from_rank_response(object(), version)
