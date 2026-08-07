from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from inspect import signature
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

import bongard.closed_visual_predicates as closed
import bongard.composite_visual_packet as composite
from bongard.closed_visual_predicates import (
    SYMMETRY_THRESHOLDS_PPM,
    ClosedPanelPredicate,
    ClosedPredicateResult,
    DirectCountPredicate,
    FrozenClosedPredicateLibrary,
    OracleDiagnosis,
    SupportExpressibilityResult,
    SymmetryMetric,
    SymmetryThresholdPredicate,
    closed_visual_predicate_source_digest,
    evaluate_closed_predicate,
    freeze_closed_predicate_library,
    freeze_complete_closed_predicate_library,
    support_only_expressibility_oracle,
    verify_closed_predicate_result,
)
from bongard.composite_visual_packet import (
    BilateralSymmetryWitnessPacket,
    ExactPanelWitnessPacket,
    PpmInterval,
    extract_exact_panel_witness_packet,
)
from bongard.evidence import Disposition
from bongard.relational_visual_query import (
    PointContactClause,
    Rational,
    RelationalVisualQuery,
)
from bongard.typed_visual_proposal import TypedDeterministicAtom
from bongard.visual_predicate_catalog import direct_visual_catalog_digest


def _png(panel: np.ndarray) -> bytes:
    encoded = BytesIO()
    Image.fromarray(panel, mode="L").save(encoded, format="PNG")
    return encoded.getvalue()


@lru_cache(maxsize=None)
def _symmetric_packet() -> ExactPanelWitnessPacket:
    panel = np.full((64, 64), 255, dtype=np.uint8)
    panel[12:52, 12:52] = 0
    panel[18:46, 18:46] = 255
    return extract_exact_panel_witness_packet(_png(panel))


@lru_cache(maxsize=None)
def _asymmetric_packet() -> ExactPanelWitnessPacket:
    panel = np.full((64, 64), 255, dtype=np.uint8)
    panel[15:45, 15:22] = 0
    panel[37:46, 22:52] = 0
    panel[20:27, 38:51] = 0
    return extract_exact_panel_witness_packet(_png(panel))


@lru_cache(maxsize=None)
def _two_loop_packet() -> ExactPanelWitnessPacket:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    panel[10:35, 10:35] = 0
    panel[15:30, 15:30] = 255
    panel[45:88, 45:88] = 0
    panel[50:83, 50:83] = 255
    return extract_exact_panel_witness_packet(_png(panel))


def _direct_atom(atom_id: str, key: str, count: int) -> TypedDeterministicAtom:
    return TypedDeterministicAtom(
        atom_id=atom_id,
        catalog_key=key,
        comparison="equal",
        arguments=(("target_count", count),),
    )


def _replace_symmetry_scenarios(
    packet: ExactPanelWitnessPacket,
    specifications: tuple[
        tuple[Disposition, PpmInterval | None, str | None],
        tuple[Disposition, PpmInterval | None, str | None],
        tuple[Disposition, PpmInterval | None, str | None],
    ],
) -> ExactPanelWitnessPacket:
    original = packet.bilateral_symmetry
    scenarios = []
    for old, (disposition, interval, message) in zip(
        original.scenarios, specifications, strict=True
    ):
        scenarios.append(
            composite._make_scenario_witness(
                scenario_id=old.scenario_id,
                foreground_strength_threshold=old.foreground_strength_threshold,
                morphology=old.morphology,
                panel_digest=old.panel_digest,
                source_mask_digest=old.source_mask_digest,
                disposition=disposition,
                coverage_ppm=interval,
                best_axis_millidegrees=(
                    90_000 if disposition is Disposition.PRESENT else None
                ),
                foreground_pixels=(
                    100 if disposition is Disposition.PRESENT else None
                ),
                reason=(
                    message
                    if disposition in {Disposition.INDETERMINATE, Disposition.ERROR}
                    else None
                ),
                certificate=(
                    message
                    if disposition is Disposition.CERTIFIED_ABSENT
                    else None
                ),
                error_type=(
                    "SyntheticExtractionError"
                    if disposition is Disposition.ERROR
                    else None
                ),
                extractor_artifact_digest=old.extractor_artifact_digest,
            )
        )
    bilateral = BilateralSymmetryWitnessPacket(
        panel_digest=original.panel_digest,
        width_pixels=original.width_pixels,
        height_pixels=original.height_pixels,
        parent_visual_bundle_digest=original.parent_visual_bundle_digest,
        extractor_source_digest=original.extractor_source_digest,
        base_visual_extractor_digest=original.base_visual_extractor_digest,
        bilateral_operation_digest=original.bilateral_operation_digest,
        extractor_artifact_digest=original.extractor_artifact_digest,
        scenarios=tuple(scenarios),
    )
    return replace(packet, bilateral_symmetry=bilateral)


def test_tagged_union_executes_relational_direct_and_symmetry_variants() -> None:
    packet = _two_loop_packet()
    relational = ClosedPanelPredicate.relational(
        RelationalVisualQuery.factorized_shape_ratio(
            numerator_side_count=4,
            denominator_side_count=4,
            ratio=Rational(1, 2),
        )
    )
    direct = ClosedPanelPredicate.direct(
        DirectCountPredicate(
            (_direct_atom("atom-00", "component.count", 2),),
            direct_visual_catalog_digest(),
        )
    )
    symmetry = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )

    for predicate in (relational, direct, symmetry):
        result = evaluate_closed_predicate(predicate, packet)
        assert result.disposition is Disposition.PRESENT
        assert ClosedPanelPredicate.from_data(predicate.to_data()) == predicate
        assert ClosedPredicateResult.from_data(result.to_data()) == result
        assert verify_closed_predicate_result(result, predicate, packet) == result


def test_symmetry_intervals_use_positive_coverage_and_positive_residual() -> None:
    packet = _replace_symmetry_scenarios(
        _symmetric_packet(),
        (
            (Disposition.PRESENT, PpmInterval(850_000, 950_000), None),
            (Disposition.PRESENT, PpmInterval(850_000, 950_000), None),
            (Disposition.PRESENT, PpmInterval(850_000, 950_000), None),
        ),
    )
    coverage = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )
    assert evaluate_closed_predicate(coverage, packet).disposition is (
        Disposition.INDETERMINATE
    )

    residual_packet = _replace_symmetry_scenarios(
        _symmetric_packet(),
        (
            (Disposition.PRESENT, PpmInterval(600_000, 700_000), None),
            (Disposition.PRESENT, PpmInterval(600_000, 700_000), None),
            (Disposition.PRESENT, PpmInterval(600_000, 700_000), None),
        ),
    )
    mismatch = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.REFLECTION_MISMATCH_AT_LEAST, 250_000
        )
    )
    assert residual_packet.bilateral_symmetry.scenarios[0].mismatch_ppm == (
        PpmInterval(300_000, 400_000)
    )
    assert evaluate_closed_predicate(mismatch, residual_packet).disposition is (
        Disposition.PRESENT
    )


def test_cross_scenario_disagreement_and_extraction_error_propagate() -> None:
    mixed = _replace_symmetry_scenarios(
        _symmetric_packet(),
        (
            (Disposition.PRESENT, PpmInterval(950_000, 950_000), None),
            (Disposition.PRESENT, PpmInterval(700_000, 700_000), None),
            (Disposition.PRESENT, PpmInterval(950_000, 950_000), None),
        ),
    )
    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )
    mixed_result = evaluate_closed_predicate(predicate, mixed)
    assert tuple(item.disposition for item in mixed_result.scenarios) == (
        Disposition.PRESENT,
        Disposition.CERTIFIED_ABSENT,
        Disposition.PRESENT,
    )
    assert mixed_result.disposition is Disposition.INDETERMINATE

    errored = _replace_symmetry_scenarios(
        _symmetric_packet(),
        (
            (Disposition.ERROR, None, "failed extraction"),
            (Disposition.ERROR, None, "failed extraction"),
            (Disposition.ERROR, None, "failed extraction"),
        ),
    )
    error_result = evaluate_closed_predicate(predicate, errored)
    assert error_result.disposition is Disposition.ERROR
    assert all(
        item.disposition is not Disposition.CERTIFIED_ABSENT
        for item in error_result.scenarios
    )


def test_schema_has_no_negation_polarity_or_arbitrary_executable_escape() -> None:
    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )
    negative = deepcopy(predicate.to_data())
    negative["polarity"] = "negative"
    with pytest.raises(ValueError, match="fields differ"):
        ClosedPanelPredicate.from_data(negative)

    logical_not = deepcopy(predicate.to_data())
    logical_not["kind"] = "not"
    with pytest.raises(ValueError):
        ClosedPanelPredicate.from_data(logical_not)

    reversed_comparison = deepcopy(predicate.to_data())
    reversed_comparison["payload"]["comparison"] = "at_most"
    with pytest.raises(ValueError, match="orientation"):
        ClosedPanelPredicate.from_data(reversed_comparison)

    callback = deepcopy(predicate.to_data())
    callback["payload"]["callback"] = "lambda panel: True"
    with pytest.raises(ValueError, match="fields differ"):
        ClosedPanelPredicate.from_data(callback)


def test_support_oracle_distinguishes_model_miss_from_language_hole() -> None:
    low = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 250_000
        )
    )
    separator = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )
    # Freeze first.  Only after this immutable digest exists do support packets
    # enter the oracle call.
    library = freeze_closed_predicate_library((low, separator))
    frozen_digest = library.digest
    result = support_only_expressibility_oracle(
        library,
        positive_support_packets=(_symmetric_packet(),),
        negative_support_packets=(_asymmetric_packet(),),
        model_predicate=low,
    )

    assert library.digest == frozen_digest
    assert result.diagnosis is OracleDiagnosis.MODEL_MISSED_SEPARATOR
    assert result.separator_digests == (separator.digest,)
    assert result.model_is_exact_separator is False
    assert SupportExpressibilityResult.from_data(result.to_data()) == result

    no_separator_library = freeze_closed_predicate_library((low,))
    no_separator = support_only_expressibility_oracle(
        no_separator_library,
        positive_support_packets=(_symmetric_packet(),),
        negative_support_packets=(_asymmetric_packet(),),
    )
    assert no_separator.diagnosis is OracleDiagnosis.NO_LANGUAGE_SEPARATOR
    assert no_separator.separator_digests == ()


def test_library_freeze_boundaries_current_source_and_oracle_model_iff() -> None:
    assert tuple(signature(freeze_complete_closed_predicate_library).parameters) == ()
    assert tuple(signature(freeze_closed_predicate_library).parameters) == (
        "predicates",
    )
    assert all(
        "packet" not in name and "label" not in name and "model" not in name
        for function in (
            freeze_complete_closed_predicate_library,
            freeze_closed_predicate_library,
        )
        for name in signature(function).parameters
    )

    member = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, SYMMETRY_THRESHOLDS_PPM[0]
        )
    )
    library = freeze_closed_predicate_library((member,))
    assert library.source_digest == closed_visual_predicate_source_digest()
    with pytest.raises(ValueError, match="source identity is not current"):
        FrozenClosedPredicateLibrary(
            construction_id=library.construction_id,
            source_digest="0" * 64,
            evaluator_digest=library.evaluator_digest,
            members=library.members,
        )

    valid = support_only_expressibility_oracle(
        library,
        positive_support_packets=(_symmetric_packet(),),
        negative_support_packets=(_asymmetric_packet(),),
        model_predicate=member,
    )
    missing_boolean = deepcopy(valid.to_data())
    missing_boolean["model_is_exact_separator"] = None
    with pytest.raises(ValueError, match="requires an exact-separator result"):
        SupportExpressibilityResult.from_data(missing_boolean)

    stray_boolean = deepcopy(valid.to_data())
    stray_boolean["model_predicate_digest"] = None
    with pytest.raises(ValueError, match="absent model predicate"):
        SupportExpressibilityResult.from_data(stray_boolean)


def test_complete_library_has_exact_deterministic_member_count_and_order() -> None:
    identity = closed.complete_closed_predicate_library_identity()
    library = freeze_complete_closed_predicate_library()
    members = library.members
    # 1,260 proposer-reachable contact-disabled relational predicates +
    # (10*8 + C(10,2)*8^2 + C(10,3)*8^3) direct conjunctions + 2*9 symmetry.
    assert len(members) == 65_678
    assert identity.member_count == len(members)
    assert identity.construction_id == (
        "complete-proposer-reachable-closed-union/v2"
    )
    assert identity.construction_id == library.construction_id
    assert identity.source_digest == library.source_digest
    assert identity.evaluator_digest == library.evaluator_digest
    assert len(identity.construction_grid_digest) == 64
    assert len(identity.complete_member_digest) == 64
    digests = tuple(item.digest for item in members)
    assert digests == tuple(sorted(digests))
    assert len(digests) == len(set(digests))
    relational = tuple(
        item for item in members if item.kind is closed.ClosedPredicateKind.RELATIONAL
    )
    assert len(relational) == 1_260
    assert all(
        not any(
            isinstance(clause, PointContactClause)
            for clause in item.payload.clauses
        )
        for item in relational
    )

    same = FrozenClosedPredicateLibrary(
        construction_id=library.construction_id,
        source_digest=library.source_digest,
        evaluator_digest=library.evaluator_digest,
        members=tuple(members),
    )
    assert same.digest == library.digest


def test_oracle_disposition_kernel_matches_full_results_for_complete_library() -> None:
    members = freeze_complete_closed_predicate_library().members
    assert len(members) == 65_678
    direct_cache: closed.DirectAtomCache = {}
    atom_digest_cache: closed.DirectAtomDigestCache = {}

    for packet in (_symmetric_packet(), _asymmetric_packet()):
        packet_digest = packet.digest()
        for index, predicate in enumerate(members):
            fast_scenarios, fast_panel = closed._evaluate_closed_dispositions(
                predicate,
                packet,
                direct_atom_cache=direct_cache,
                direct_atom_digest_cache=atom_digest_cache,
                precomputed_packet_digest=packet_digest,
            )
            full = closed._evaluate_closed_predicate(
                predicate,
                packet,
                direct_atom_cache=direct_cache,
                direct_atom_digest_cache=atom_digest_cache,
                packet_is_prevalidated=True,
                precomputed_packet_digest=packet_digest,
            )
            full_scenarios = tuple(item.disposition for item in full.scenarios)
            if (fast_scenarios, fast_panel) != (
                full_scenarios,
                full.disposition,
            ):
                pytest.fail(
                    "oracle disposition kernel diverged at "
                    f"member {index} ({predicate.digest})"
                )


def test_result_and_library_digest_tampering_is_rejected() -> None:
    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST, 900_000
        )
    )
    result = evaluate_closed_predicate(predicate, _symmetric_packet())
    tampered = deepcopy(result.to_data())
    tampered["scenarios"][0]["disposition"] = "certified_absent"
    with pytest.raises(ValueError, match="scenario consensus"):
        ClosedPredicateResult.from_data(tampered)

    library = freeze_closed_predicate_library((predicate,))
    library_data = deepcopy(library.to_data())
    library_data["members"][0]["payload"]["threshold_ppm"] = 123_456
    with pytest.raises(ValueError, match="outside the frozen grid"):
        FrozenClosedPredicateLibrary.from_data(library_data)
