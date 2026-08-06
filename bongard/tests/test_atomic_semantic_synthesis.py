from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re

import pytest

import bongard.atomic_semantic_synthesis as atomic_module
from bongard.artifacts import TruthEvidenceRecord, canonical_digest, canonical_json
from bongard.atomic_semantic_synthesis import (
    CALIBRATED_SELECTION_SCOPE,
    OPERATIONAL_SELECTION_SCOPE,
    AtomicEvidenceBinding,
    AtomicSelectionArchive,
    AtomicSemanticSynthesisError,
    AtomicSoftPredicate,
    AtomicSupportCell,
    AtomicSupportMatrix,
    NoExactSeparatorError,
    NoExactSeparatorDiagnostics,
    OperationalNonmatchRecord,
    PanelDescriptionBinding,
    cold_decode_and_recompute_no_exact_separator,
    cold_decode_and_replay_atomic_selection,
    evaluate_atomic_formula,
    synthesize_atomic_conjunction,
)
from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty


SOURCE_PROPOSAL = canonical_digest({"fixture": "source proposal"})
SCORER_PROTOCOL = canonical_digest({"fixture": "atomic scorer protocol"})
DESCRIPTION_PROTOCOL = canonical_digest({"fixture": "description protocol"})
RUN_COMMITMENT = canonical_digest({"fixture": "run commitment"})
CALIBRATION = canonical_digest({"fixture": "externally pinned calibration"})
PANEL_IDS = ("negative-a", "negative-b", "positive-a", "positive-b")
LABELS = {
    "negative-a": False,
    "negative-b": False,
    "positive-a": True,
    "positive-b": True,
}


def _bindings(panel_ids: tuple[str, ...] = PANEL_IDS):
    return tuple(
        PanelDescriptionBinding.create(
            panel_id,
            hashlib.sha256(f"pixels:{panel_id}".encode()).hexdigest(),
            "A simple angular figure is centered on a plain background.",
            phase="support",
            description_protocol_digest=DESCRIPTION_PROTOCOL,
            validated_receipt_digest=hashlib.sha256(
                f"description-receipt:{panel_id}".encode()
            ).hexdigest(),
            run_commitment_digest=RUN_COMMITMENT,
            call_ordinal=1 + panel_ids.index(panel_id),
        )
        for panel_id in panel_ids
    )


def _atom(
    positive_description: str,
    cue_description: str,
    *,
    panel_ids: tuple[str, ...] = PANEL_IDS,
) -> AtomicSoftPredicate:
    return AtomicSoftPredicate.create(
        source_proposal_digest=SOURCE_PROPOSAL,
        scorer_protocol_digest=SCORER_PROTOCOL,
        positive_description=positive_description,
        cue_description=positive_description,
        panel_descriptions=_bindings(panel_ids),
    )


def _binding(
    atom: AtomicSoftPredicate,
    panel_id: str,
    *,
    scope: str = OPERATIONAL_SELECTION_SCOPE,
) -> AtomicEvidenceBinding:
    panel = atom.panel_binding(panel_id)
    ordinal = 20 + tuple(item.panel_id for item in atom.panel_descriptions).index(
        panel_id
    )
    return AtomicEvidenceBinding(
        atom_id=atom.atom_id,
        panel_digest=panel.panel_digest,
        panel_description_digest=panel.description_digest,
        scorer_protocol_digest=atom.scorer_protocol_digest,
        run_commitment_digest=panel.run_commitment_digest,
        scorer_producer="fixture-atomic-scorer",
        scorer_version="1",
        scorer_method=(
            "calibrated-one-phrase-score"
            if scope == CALIBRATED_SELECTION_SCOPE
            else "operational-one-phrase-score"
        ),
        scorer_run_id="run-fixture",
        scorer_receipt_digest=hashlib.sha256(
            f"scorer-receipt:{panel_id}:{scope}".encode()
        ).hexdigest(),
        scorer_output_digest=hashlib.sha256(
            f"scorer-output:{panel_id}:{scope}".encode()
        ).hexdigest(),
        scorer_call_digest=hashlib.sha256(
            f"scorer-call:{panel_id}:{scope}".encode()
        ).hexdigest(),
        scorer_call_ordinal=ordinal,
        observation_scope=scope,
        calibration_digest=(
            CALIBRATION if scope == CALIBRATED_SELECTION_SCOPE else None
        ),
    )


def _provenance(
    atom: AtomicSoftPredicate,
    panel_id: str,
    *,
    scope: str = OPERATIONAL_SELECTION_SCOPE,
) -> Provenance:
    binding = _binding(atom, panel_id, scope=scope)
    return Provenance(
        producer=binding.scorer_producer,
        version=binding.scorer_version,
        method=binding.scorer_method,
        input_digests=binding.input_digests,
        artifact_digest=binding.scorer_receipt_digest,
        run_id=binding.scorer_run_id,
        details=binding.provenance_details,
    )


def _query_binding(
    atom_id: str,
    *,
    panel_digest: str | None = None,
    scope: str = OPERATIONAL_SELECTION_SCOPE,
) -> AtomicEvidenceBinding:
    return AtomicEvidenceBinding(
        atom_id=atom_id,
        panel_digest=panel_digest or hashlib.sha256(b"query pixels").hexdigest(),
        panel_description_digest=hashlib.sha256(
            b"frozen query description"
        ).hexdigest(),
        scorer_protocol_digest=SCORER_PROTOCOL,
        run_commitment_digest=RUN_COMMITMENT,
        scorer_producer="fixture-query-scorer",
        scorer_version="1",
        scorer_method=(
            "calibrated-query-score"
            if scope == CALIBRATED_SELECTION_SCOPE
            else "operational-query-score"
        ),
        scorer_run_id="run-fixture",
        scorer_receipt_digest=hashlib.sha256(
            f"query-receipt:{scope}".encode()
        ).hexdigest(),
        scorer_output_digest=hashlib.sha256(
            f"query-output:{scope}".encode()
        ).hexdigest(),
        scorer_call_digest=hashlib.sha256(
            f"query-call:{scope}".encode()
        ).hexdigest(),
        scorer_call_ordinal=40,
        observation_scope=scope,
        calibration_digest=(
            CALIBRATION if scope == CALIBRATED_SELECTION_SCOPE else None
        ),
    )


def _bound_provenance(binding: AtomicEvidenceBinding, method: str | None = None):
    return Provenance(
        binding.scorer_producer,
        binding.scorer_version,
        method or binding.scorer_method,
        input_digests=binding.input_digests,
        artifact_digest=binding.scorer_receipt_digest,
        run_id=binding.scorer_run_id,
        details=binding.provenance_details,
    )


def _evidence(
    atom: AtomicSoftPredicate,
    panel_id: str,
    disposition: str,
    *,
    scope: str = OPERATIONAL_SELECTION_SCOPE,
) -> Evidence[bool]:
    provenance = _provenance(atom, panel_id, scope=scope)
    if disposition == "present":
        return Evidence.present(True, provenance)
    if disposition == "absent":
        return Evidence.operational_nonmatch(
            provenance, "the fixed observer phrase did not match"
        )
    if disposition == "operational_nonmatch":
        return Evidence.operational_nonmatch(
            provenance, "the fixed observer phrase did not match"
        )
    if disposition == "indeterminate":
        return Evidence.indeterminate(
            provenance, "calibrated interval straddles the affirmative boundary"
        )
    if disposition == "error":
        return Evidence.error(
            provenance, "FitFailure", "the scorer fit failed on this panel"
        )
    raise AssertionError(disposition)


def _matrix(
    atoms: tuple[AtomicSoftPredicate, ...],
    outcomes: dict[str, dict[str, str]],
    *,
    reverse_inputs: bool = False,
    scope: str = OPERATIONAL_SELECTION_SCOPE,
) -> AtomicSupportMatrix:
    cells = [
        AtomicSupportCell.capture(
            atom,
            panel_id,
            _evidence(atom, panel_id, state, scope=scope),
            evidence_binding=_binding(atom, panel_id, scope=scope),
        )
        for atom in atoms
        for panel_id, state in outcomes[atom.atom_id].items()
    ]
    return AtomicSupportMatrix.create(
        tuple(reversed(atoms)) if reverse_inputs else atoms,
        tuple(reversed(cells)) if reverse_inputs else cells,
    )


def _near_miss_matrix(*, reverse_inputs: bool = False):
    left = _atom(
        "The figure has a pointed front end.",
        "A pointed front tip is visible.",
    )
    right = _atom(
        "The figure has an oblique rear edge.",
        "A rear edge meets the body obliquely.",
    )
    outcomes = {
        left.atom_id: {
            "negative-a": "absent",
            "negative-b": "present",
            "positive-a": "present",
            "positive-b": "present",
        },
        right.atom_id: {
            "negative-a": "present",
            "negative-b": "absent",
            "positive-a": "present",
            "positive-b": "present",
        },
    }
    return (left, right), _matrix(
        (left, right), outcomes, reverse_inputs=reverse_inputs
    )


@pytest.mark.parametrize(
    "text",
    [
        "The object is bird-like or fish-like.",
        "Either end can be the pointed front.",
        "One of the two loops is large.",
        "One-of the two loops is large.",
        "One / of the two loops is large.",
        "One\u2011of the two loops is large.",
        "The outline is bird-like/fish-like.",
        "Alternately, the front can point left.",
        "Alternatively the outline can be rounded.",
        "The object is EITHER narrow or broad.",
    ],
)
def test_atomic_predicate_rejects_embedded_disjunction(text: str) -> None:
    with pytest.raises(AtomicSemanticSynthesisError, match="non-atomic"):
        _atom(text, "A pointed end is visible.")


def test_atomic_lexical_guard_conservatively_rejects_conjunction() -> None:
    with pytest.raises(AtomicSemanticSynthesisError, match="non-atomic"):
        _atom("The circle and square touch.", "ignored")
    atom = _atom("The circle touches the square.", "ignored")
    assert atom.positive_description == "The circle touches the square."


@pytest.mark.parametrize(
    "text",
    [
        "The figure is circle-free.",
        "The figure avoids circles.",
        "The outline fails to contain a circle.",
        "The figure is circular, otherwise it is square.",
        "The figure is circular; failing that, it is square.",
        "The figure has whichever of a circle and a square is visible.",
        "The figure contains any among a circle and a square.",
        "The figure is bird-like and has oblique angles.",
        "The figure is circular versus square.",
    ],
)
def test_surface_atomic_guard_rejects_reviewed_laundering(text: str) -> None:
    with pytest.raises(AtomicSemanticSynthesisError, match="non-atomic"):
        _atom(text, "ignored")


def test_panel_description_binding_attests_pixels_phase_receipt_run_and_call() -> None:
    original = _bindings(("panel-a",))[0]
    changed_pixels = PanelDescriptionBinding.create(
        "panel-a",
        hashlib.sha256(b"different pixels").hexdigest(),
        original.description,
        phase="support",
        description_protocol_digest=DESCRIPTION_PROTOCOL,
        validated_receipt_digest=original.validated_receipt_digest,
        run_commitment_digest=RUN_COMMITMENT,
        call_ordinal=original.call_ordinal,
    )
    assert original.description_digest != changed_pixels.description_digest
    assert PanelDescriptionBinding.from_data(original.to_data()) == original

    leaked = dict(original.content_data())
    leaked["description"] = "This is the positive support panel."
    leaked["description_digest"] = canonical_digest(leaked)
    with pytest.raises(AtomicSemanticSynthesisError, match="role leak"):
        PanelDescriptionBinding.from_data(leaked)

    query = PanelDescriptionBinding.create(
        "panel-a",
        original.panel_digest,
        original.description,
        phase="query",
        description_protocol_digest=DESCRIPTION_PROTOCOL,
        validated_receipt_digest=original.validated_receipt_digest,
        run_commitment_digest=RUN_COMMITMENT,
        call_ordinal=30,
    )
    with pytest.raises(AtomicSemanticSynthesisError, match="support descriptions only"):
        AtomicSoftPredicate.create(
            source_proposal_digest=SOURCE_PROPOSAL,
            scorer_protocol_digest=SCORER_PROTOCOL,
            positive_description="The figure is pointed.",
            cue_description="The figure is pointed.",
            panel_descriptions=(query,),
        )


def test_atomic_predicate_binds_proposal_descriptions_and_scorer() -> None:
    atom = _atom(
        "The outline resembles a bird-like object.",
        "A beak-like pointed front is visible.",
    )

    assert atom.atom_id == canonical_digest(atom.content_data())
    assert atom.source_proposal_digest == SOURCE_PROPOSAL
    assert atom.scorer_protocol_digest == SCORER_PROTOCOL
    assert atom.panel_descriptions_digest == canonical_digest(
        [item.to_data() for item in atom.panel_descriptions]
    )
    assert atom.to_data()["cue_description"] == atom.positive_description
    assert "cues" not in atom.to_data()
    assert AtomicSoftPredicate.from_data(atom.to_data()) == atom

    with pytest.raises(AtomicSemanticSynthesisError, match="same single"):
        AtomicSoftPredicate.create(
            source_proposal_digest=SOURCE_PROPOSAL,
            scorer_protocol_digest=SCORER_PROTOCOL,
            positive_description="The outline resembles a bird-like object.",
            cue_description="A pointed front is visible.",
            panel_descriptions=_bindings(),
        )


def test_support_matrix_requires_the_exact_cartesian_product() -> None:
    first = _atom("The figure is pointed.", "A pointed tip is visible.")
    second = _atom("The figure is rounded.", "A rounded arc is visible.")
    outcomes = {
        atom.atom_id: {panel_id: "present" for panel_id in PANEL_IDS}
        for atom in (first, second)
    }
    matrix = _matrix((first, second), outcomes)

    assert len(matrix.cells) == len(matrix.atoms) * len(matrix.panel_ids) == 8
    assert tuple((cell.atom_id, cell.panel_id) for cell in matrix.cells) == tuple(
        (atom.atom_id, panel_id)
        for atom in matrix.atoms
        for panel_id in matrix.panel_ids
    )
    with pytest.raises(AtomicSemanticSynthesisError, match="Cartesian"):
        AtomicSupportMatrix.create(matrix.atoms, matrix.cells[:-1])
    with pytest.raises(AtomicSemanticSynthesisError, match="Cartesian"):
        AtomicSupportMatrix.create(matrix.atoms, (*matrix.cells, matrix.cells[-1]))

    with pytest.raises(ValueError, match="only present True"):
        AtomicSupportCell.capture(
            first,
            "positive-a",
            Evidence.present(False, _provenance(first, "positive-a")),
            evidence_binding=_binding(first, "positive-a"),
        )


def test_support_cell_capture_and_decode_require_exact_provenance_binding() -> None:
    atom = _atom("The figure is pointed.", "A pointed tip is visible.")
    panel_id = "positive-a"
    binding = _binding(atom, panel_id)
    wrong = Provenance(
        producer="fixture-atomic-scorer",
        version="1",
        method="frozen-one-cue-score",
        input_digests=(
            binding.panel_digest,
            binding.atom_id,
            binding.panel_description_digest,
            binding.scorer_protocol_digest,
        ),
    )
    with pytest.raises(AtomicSemanticSynthesisError, match="canonical order"):
        AtomicSupportCell.capture(
            atom,
            panel_id,
            Evidence.present(True, wrong),
            evidence_binding=binding,
        )

    cell = AtomicSupportCell.capture(
        atom,
        panel_id,
        _evidence(atom, panel_id, "present"),
        evidence_binding=binding,
    )
    rehashed = copy.deepcopy(cell.to_data())
    rehashed["evidence"]["provenance"]["input_digests"] = list(
        reversed(binding.input_digests)
    )
    rehashed["evidence_digest"] = canonical_digest(rehashed["evidence"])
    rehashed_content = dict(rehashed)
    rehashed_content.pop("cell_digest")
    rehashed["cell_digest"] = canonical_digest(rehashed_content)

    with pytest.raises(AtomicSemanticSynthesisError, match="canonical order"):
        AtomicSupportCell.from_data(rehashed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("producer", "replacement-scorer"),
        ("version", "2"),
        ("method", "replacement-method"),
        ("run_id", "replacement-run"),
        ("artifact_digest", "f" * 64),
        ("details", (("call_digest", "f" * 64),)),
    ],
)
def test_scorer_identity_receipt_output_run_and_ordinal_are_authorized(
    field: str, replacement: object
) -> None:
    atom = _atom("The figure is pointed.", "ignored")
    panel_id = "positive-a"
    binding = _binding(atom, panel_id)
    values = {
        "producer": binding.scorer_producer,
        "version": binding.scorer_version,
        "method": binding.scorer_method,
        "input_digests": binding.input_digests,
        "artifact_digest": binding.scorer_receipt_digest,
        "run_id": binding.scorer_run_id,
        "details": binding.provenance_details,
    }
    values[field] = replacement
    provenance = Provenance(**values)  # type: ignore[arg-type]
    with pytest.raises(AtomicSemanticSynthesisError, match="identity differs"):
        AtomicSupportCell.capture(
            atom,
            panel_id,
            Evidence.present(True, provenance),
            evidence_binding=binding,
        )


def test_exact_synthesis_finds_conjunctive_near_miss_separator() -> None:
    atoms, matrix = _near_miss_matrix()

    archive = synthesize_atomic_conjunction(
        matrix,
        LABELS,
        max_atoms=2,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )

    assert archive.formula == {
        "kind": "all",
        "atom_ids": sorted(atom.atom_id for atom in atoms),
    }
    assert len(archive.selected_atom_ids) == 2
    assert all(panel_ids for _, panel_ids in archive.load_bearing)
    replay = dict(archive.replay_support())
    assert isinstance(replay["negative-a"], OperationalNonmatchRecord)
    assert isinstance(replay["negative-b"], OperationalNonmatchRecord)
    assert replay["positive-a"].disposition is Disposition.PRESENT
    assert replay["positive-b"].disposition is Disposition.PRESENT


def test_operational_nonmatch_has_distinct_archive_scope_and_no_truth_absence() -> None:
    left = _atom("The figure has a pointed front.", "ignored")
    right = _atom("The figure has an oblique rear edge.", "ignored")
    outcomes = {
        left.atom_id: {
            "negative-a": "operational_nonmatch",
            "negative-b": "present",
            "positive-a": "present",
            "positive-b": "present",
        },
        right.atom_id: {
            "negative-a": "present",
            "negative-b": "operational_nonmatch",
            "positive-a": "present",
            "positive-b": "present",
        },
    }
    matrix = _matrix(
        (left, right), outcomes, scope=OPERATIONAL_SELECTION_SCOPE
    )
    archive = synthesize_atomic_conjunction(
        matrix,
        LABELS,
        max_atoms=2,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )

    assert archive.selection_scope == OPERATIONAL_SELECTION_SCOPE
    assert archive.to_data()["claim_authority"] == {
        "calibration_authorized": False,
        "benchmark_claim_authorized": False,
        "semantic_truth_claim": False,
    }
    replay = dict(archive.replay_support())
    assert isinstance(replay["negative-a"], OperationalNonmatchRecord)
    assert replay["negative-a"].disposition is Disposition.INDETERMINATE
    assert replay["negative-a"].to_data()["disposition"] == (
        "operational_nonmatch"
    )
    assert AtomicSelectionArchive.from_data(archive.to_data()) == archive
    assert cold_decode_and_replay_atomic_selection(
        archive.to_data(), expected_archive_digest=archive.archive_digest
    ) == archive.replay_support()

    with pytest.raises(AtomicSemanticSynthesisError, match="disabled"):
        synthesize_atomic_conjunction(
            matrix,
            LABELS,
            selection_scope=CALIBRATED_SELECTION_SCOPE,
        )


def test_operational_authorization_cannot_launder_certified_absence() -> None:
    atom = _atom("The figure is pointed.", "ignored")
    binding = _binding(
        atom, "negative-a", scope=OPERATIONAL_SELECTION_SCOPE
    )
    provenance = _bound_provenance(binding)
    with pytest.raises(AtomicSemanticSynthesisError, match="disabled"):
        AtomicSupportCell.capture(
            atom,
            "negative-a",
            Evidence.certified_absent(provenance, "fabricated certificate"),
            evidence_binding=binding,
        )

    calibrated = _binding(
        atom, "negative-a", scope=CALIBRATED_SELECTION_SCOPE
    )
    with pytest.raises(
        AtomicSemanticSynthesisError, match="operational observer authorization"
    ):
        AtomicSupportCell.capture(
            atom,
            "negative-a",
            Evidence.operational_nonmatch(
                _bound_provenance(calibrated), "uncalibrated mismatch"
            ),
            evidence_binding=calibrated,
        )

    with pytest.raises(AtomicSemanticSynthesisError, match="disabled"):
        AtomicSupportCell.capture(
            atom,
            "negative-a",
            Evidence.certified_absent(
                _bound_provenance(calibrated), "fabricated calibrated absence"
            ),
            evidence_binding=calibrated,
        )


def test_selection_tie_break_is_mdl_then_digest_and_input_order_independent() -> None:
    short = _atom("The figure is pointed.", "A pointed tip is visible.")
    long = _atom(
        "The figure has a clearly visible strongly pointed front end.",
        "A clearly delineated pointed front tip is visible on the outline.",
    )
    outcomes = {
        atom.atom_id: {
            panel_id: "present" if LABELS[panel_id] else "absent"
            for panel_id in PANEL_IDS
        }
        for atom in (short, long)
    }
    forward = _matrix((short, long), outcomes)
    reverse = _matrix((short, long), outcomes, reverse_inputs=True)

    first = synthesize_atomic_conjunction(
        forward, LABELS, selection_scope=OPERATIONAL_SELECTION_SCOPE
    )
    second = synthesize_atomic_conjunction(
        reverse,
        dict(reversed(tuple(LABELS.items()))),
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )

    assert first.selected_atom_ids == (short.atom_id,)
    assert first.matrix.matrix_digest == second.matrix.matrix_digest
    assert first.archive_digest == second.archive_digest

    arc = _atom("The figure shows arc ink.", "Visible arc ink forms a mark.")
    dot = _atom("The figure shows dot ink.", "Visible dot ink forms a mark.")
    assert arc.description_utf8_bytes == dot.description_utf8_bytes
    tied_outcomes = {
        atom.atom_id: {
            panel_id: "present" if LABELS[panel_id] else "absent"
            for panel_id in PANEL_IDS
        }
        for atom in (arc, dot)
    }
    tied = synthesize_atomic_conjunction(
        _matrix((dot, arc), tied_outcomes, reverse_inputs=True),
        LABELS,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )
    assert tied.selected_atom_ids == (min(arc.atom_id, dot.atom_id),)


def test_reverse_atom_cannot_be_rescued_by_negation() -> None:
    reverse = _atom(
        "The figure has a bird-like outline.",
        "A beak-like pointed front is visible.",
    )
    outcomes = {
        reverse.atom_id: {
            panel_id: "absent" if LABELS[panel_id] else "present"
            for panel_id in PANEL_IDS
        }
    }

    with pytest.raises(NoExactSeparatorError) as caught:
        synthesize_atomic_conjunction(
            _matrix((reverse,), outcomes),
            LABELS,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )

    diagnostic = caught.value.diagnostics
    assert diagnostic.reason == (
        "no atom is total and present on every positive support panel"
    )
    assert diagnostic.atom_diagnostics[0].eligible is False
    assert any(
        reason.startswith("positive_not_present:")
        for reason in diagnostic.atom_diagnostics[0].rejection_reasons
    )
    assert diagnostic.best_attempt_atom_ids == ()


def test_indeterminate_and_failed_fit_are_rejected_not_read_as_negative() -> None:
    ambiguous = _atom(
        "The figure has an oblique angle.",
        "Two outline segments meet at an oblique angle.",
    )
    failed = _atom(
        "The figure has a hooked front.",
        "A hooked front segment is visible.",
    )
    outcomes = {
        ambiguous.atom_id: {
            "negative-a": "absent",
            "negative-b": "absent",
            "positive-a": "present",
            "positive-b": "indeterminate",
        },
        failed.atom_id: {
            "negative-a": "absent",
            "negative-b": "error",
            "positive-a": "present",
            "positive-b": "present",
        },
    }

    with pytest.raises(NoExactSeparatorError) as caught:
        synthesize_atomic_conjunction(
            _matrix((ambiguous, failed), outcomes),
            LABELS,
            max_atoms=2,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )

    by_atom = {
        item.atom_id: item for item in caught.value.diagnostics.atom_diagnostics
    }
    assert any(
        reason.endswith(":indeterminate")
        for reason in by_atom[ambiguous.atom_id].rejection_reasons
    )
    assert any(
        reason == "non_total:negative-b:error"
        for reason in by_atom[failed.atom_id].rejection_reasons
    )
    assert by_atom[failed.atom_id].covered_negative_panel_ids == ("negative-a",)
    assert caught.value.diagnostics.best_attempt_atom_ids == ()


def test_no_separator_diagnostic_distinguishes_small_conjunction_limit() -> None:
    first, second = (
        _atom("The figure has an angled tip.", "An angled tip is visible."),
        _atom("The figure has a curved tail.", "A curved tail is visible."),
    )
    panel_ids = ("negative-a", "negative-b", "positive-a")
    labels = {"negative-a": False, "negative-b": False, "positive-a": True}
    first = _atom(
        first.positive_description,
        first.cue_description,
        panel_ids=panel_ids,
    )
    second = _atom(
        second.positive_description,
        second.cue_description,
        panel_ids=panel_ids,
    )
    outcomes = {
        first.atom_id: {
            "negative-a": "absent",
            "negative-b": "present",
            "positive-a": "present",
        },
        second.atom_id: {
            "negative-a": "present",
            "negative-b": "absent",
            "positive-a": "present",
        },
    }

    with pytest.raises(NoExactSeparatorError) as caught:
        synthesize_atomic_conjunction(
            _matrix((first, second), outcomes),
            labels,
            max_atoms=1,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )

    diagnostic = caught.value.diagnostics
    assert diagnostic.reason == "no exact positive conjunction exists within max_atoms"
    assert diagnostic.uncovered_by_any_eligible_atom == ()
    assert len(diagnostic.best_attempt_atom_ids) == 1
    assert len(diagnostic.best_attempt_covered_negative_panel_ids) == 1
    assert diagnostic.diagnostic_digest == canonical_digest(
        diagnostic.content_data()
    )
    decoded = NoExactSeparatorDiagnostics.from_data(
        json.loads(canonical_json(diagnostic.to_data()))
    )
    assert decoded == diagnostic
    assert cold_decode_and_recompute_no_exact_separator(
        diagnostic.to_data(),
        expected_diagnostic_digest=diagnostic.diagnostic_digest,
        matrix=_matrix((first, second), outcomes),
        support_labels=labels,
        max_atoms=1,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    ) == diagnostic

    forged = copy.deepcopy(diagnostic.to_data())
    forged["positive_panel_ids"].append("positive-a")
    forged_content = dict(forged)
    forged_content.pop("diagnostic_digest")
    forged["diagnostic_digest"] = canonical_digest(forged_content)
    with pytest.raises(AtomicSemanticSynthesisError, match="panel IDs"):
        NoExactSeparatorDiagnostics.from_data(forged)

    with pytest.raises(AtomicSemanticSynthesisError, match="1..4"):
        synthesize_atomic_conjunction(
            _matrix((first, second), outcomes),
            labels,
            max_atoms=5,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )


def test_evaluator_preserves_four_dispositions_and_never_masks_error() -> None:
    atom_ids = tuple(
        sorted(
            (
                hashlib.sha256(b"a").hexdigest(),
                hashlib.sha256(b"b").hexdigest(),
            )
        )
    )
    formula = {"kind": "all", "atom_ids": list(atom_ids)}
    bindings = {
        atom_id: _query_binding(atom_id)
        for atom_id in atom_ids
    }

    def observed(
        state: str,
        atom_id: str,
        binding: AtomicEvidenceBinding | None = None,
    ) -> Evidence[bool]:
        evidence_binding = binding or bindings[atom_id]
        provenance = _bound_provenance(evidence_binding)
        if state == "present":
            return Evidence.present(True, provenance)
        if state == "nonmatch":
            return Evidence.operational_nonmatch(provenance, "frozen nonmatch")
        if state == "indeterminate":
            return Evidence.indeterminate(provenance, "frozen ambiguity")
        return Evidence.error(provenance, "FitFailure", "frozen failure")

    def disposition(first: str, second: str) -> Disposition:
        return evaluate_atomic_formula(
            formula,
            {
                atom_ids[0]: observed(first, atom_ids[0]),
                atom_ids[1]: observed(second, atom_ids[1]),
            },
            provenance_bindings=bindings,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        ).disposition

    assert disposition("present", "present") is Disposition.PRESENT
    assert disposition("present", "nonmatch") is Disposition.INDETERMINATE
    assert disposition("present", "indeterminate") is Disposition.INDETERMINATE
    assert disposition("nonmatch", "error") is Disposition.ERROR

    wrong_panel_digest = hashlib.sha256(b"different query pixels").hexdigest()
    bad_binding = {
        atom_id: _query_binding(atom_id, panel_digest=wrong_panel_digest)
        for atom_id in atom_ids
    }
    with pytest.raises(AtomicSemanticSynthesisError, match="exactly bind"):
        evaluate_atomic_formula(
            formula,
            {
                atom_id: observed("present", atom_id) for atom_id in atom_ids
            },
            provenance_bindings=bad_binding,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )

    mixed_binding = dict(bindings)
    mixed_binding[atom_ids[0]] = bad_binding[atom_ids[0]]
    with pytest.raises(AtomicSemanticSynthesisError, match="share one frozen"):
        evaluate_atomic_formula(
            formula,
            {
                atom_id: observed("present", atom_id, mixed_binding[atom_id])
                for atom_id in atom_ids
            },
            provenance_bindings=mixed_binding,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )


def test_operational_formula_nonmatch_is_not_certified_absence() -> None:
    atom_id = hashlib.sha256(b"operational atom").hexdigest()
    binding = _query_binding(atom_id, scope=OPERATIONAL_SELECTION_SCOPE)
    present = Evidence.present(True, _bound_provenance(binding))
    nonmatch = Evidence.operational_nonmatch(
        _bound_provenance(binding), "fixed phrase did not match"
    )
    formula = {"kind": "all", "atom_ids": [atom_id]}

    assert evaluate_atomic_formula(
        formula,
        {atom_id: present},
        provenance_bindings={atom_id: binding},
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    ).disposition is Disposition.PRESENT
    result = evaluate_atomic_formula(
        formula,
        {atom_id: nonmatch},
        provenance_bindings={atom_id: binding},
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )
    assert result.is_operational_nonmatch
    assert result.disposition is Disposition.INDETERMINATE
    assert result.disposition is not Disposition.CERTIFIED_ABSENT

    with pytest.raises(AtomicSemanticSynthesisError, match="disabled"):
        evaluate_atomic_formula(
            formula,
            {atom_id: nonmatch},
            provenance_bindings={atom_id: binding},
            selection_scope=CALIBRATED_SELECTION_SCOPE,
        )


@pytest.mark.parametrize(
    "uncertainty",
    [
        lambda: Uncertainty(False, 1.0),
        lambda: Uncertainty(0, 1.0),
        lambda: Uncertainty(0.0, True),
        lambda: Uncertainty(0.0, 1.0, confidence_level=1),
    ],
)
def test_uncertainty_rejects_noncanonical_bool_or_integer_numbers(
    uncertainty,
) -> None:
    with pytest.raises(ValueError, match="canonical float"):
        uncertainty()


def test_canonical_float_uncertainty_survives_atomic_cold_replay() -> None:
    atom = _atom("The figure is pointed.", "ignored", panel_ids=("n", "p"))
    outcomes = {
        atom.atom_id: {"n": "operational_nonmatch", "p": "present"}
    }
    matrix = _matrix(
        (atom,), outcomes, scope=OPERATIONAL_SELECTION_SCOPE
    )
    # Replace the two records with exact floating-point uncertainty while
    # retaining their authorized call bindings.
    cells = []
    for cell in matrix.cells:
        evidence = cell.evidence.to_evidence()
        interval = Uncertainty(0.1, 0.2, confidence_level=0.95)
        if evidence.is_operational_nonmatch:
            evidence = Evidence.operational_nonmatch(
                evidence.provenance, "fixed phrase did not match", interval
            )
        else:
            evidence = Evidence.present(True, evidence.provenance, interval)
        cells.append(
            AtomicSupportCell.capture(
                atom,
                cell.panel_id,
                evidence,
                evidence_binding=cell.evidence_binding,
            )
        )
    archive = synthesize_atomic_conjunction(
        AtomicSupportMatrix.create((atom,), cells),
        {"n": False, "p": True},
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )
    assert AtomicSelectionArchive.from_data(archive.to_data()) == archive


def test_evaluator_enforces_frozen_one_to_four_atom_bound() -> None:
    atom_ids = tuple(
        sorted(
            hashlib.sha256(f"atom:{index}".encode()).hexdigest()
            for index in range(5)
        )
    )

    with pytest.raises(AtomicSemanticSynthesisError, match="1..4 atom bound"):
        evaluate_atomic_formula(
            {"kind": "all", "atom_ids": list(atom_ids)},
            {},
            provenance_bindings={},
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )


def test_selection_archive_cold_replays_and_detects_tampering() -> None:
    _, matrix = _near_miss_matrix()
    archive = synthesize_atomic_conjunction(
        matrix,
        LABELS,
        max_atoms=2,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )
    encoded = canonical_json(archive.to_data())
    decoded_data = json.loads(encoded)
    decoded = AtomicSelectionArchive.from_data(decoded_data)

    assert decoded.archive_digest == archive.archive_digest
    assert canonical_json(decoded.to_data()) == encoded
    assert cold_decode_and_replay_atomic_selection(
        decoded_data,
        expected_archive_digest=archive.archive_digest,
    ) == decoded.replay_support()

    with pytest.raises(TypeError, match="expected_archive_digest"):
        cold_decode_and_replay_atomic_selection(decoded_data)

    rehashed = copy.deepcopy(archive.to_data())
    rehashed["max_atoms"] = 3
    rehashed_content = dict(rehashed)
    rehashed_content.pop("archive_digest")
    rehashed["archive_digest"] = canonical_digest(rehashed_content)
    assert AtomicSelectionArchive.from_data(rehashed).archive_digest != (
        archive.archive_digest
    )
    with pytest.raises(
        AtomicSemanticSynthesisError, match="expected_archive_digest"
    ):
        cold_decode_and_replay_atomic_selection(
            rehashed,
            expected_archive_digest=archive.archive_digest,
        )

    tampered = copy.deepcopy(archive.to_data())
    nonmatch_cell = next(
        cell
        for cell in tampered["matrix"]["cells"]
        if cell["evidence"]["disposition"] == "operational_nonmatch"
    )
    nonmatch_cell["evidence"]["reason"] = "fabricated replacement reason"
    with pytest.raises(ValueError, match="evidence_digest"):
        AtomicSelectionArchive.from_data(tampered)

    formula_tamper = copy.deepcopy(archive.to_data())
    formula_tamper["formula"]["atom_ids"][0] = hashlib.sha256(b"fake").hexdigest()
    with pytest.raises(ValueError):
        AtomicSelectionArchive.from_data(formula_tamper)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("atom_count", 2.0),
        ("atom_count", True),
        ("description_utf8_bytes", "not-an-integer"),
        ("description_utf8_bytes", False),
    ],
)
def test_selection_objective_requires_literal_non_bool_integers(
    field: str, replacement: object
) -> None:
    _, matrix = _near_miss_matrix()
    archive = synthesize_atomic_conjunction(
        matrix,
        LABELS,
        max_atoms=2,
        selection_scope=OPERATIONAL_SELECTION_SCOPE,
    )
    malformed = copy.deepcopy(archive.to_data())
    malformed["selection_objective"][field] = replacement

    with pytest.raises(AtomicSemanticSynthesisError, match="literal"):
        AtomicSelectionArchive.from_data(malformed)


def test_atomic_module_has_no_optional_formal_backend_dependency() -> None:
    path = Path(atomic_module.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "semantic_" + "checker" not in source.casefold()
    assert re.search(r"\b" + "lean" + r"\b", source.casefold()) is None
    assert all("semantic_" + "checker" not in item for item in imports)
    assert all(re.search(r"\b" + "lean" + r"\b", item) is None for item in imports)
    assert TruthEvidenceRecord.__module__ == "bongard.artifacts"
