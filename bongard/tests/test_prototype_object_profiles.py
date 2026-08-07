from __future__ import annotations

from copy import deepcopy

import pytest

from bongard.evidence import Disposition
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectFeatureCell,
    ObjectFeatureCellState,
    ObjectHypothesisBinding,
    ObjectLocalObservationPacket,
    ObjectProfile,
    ObjectProfileAtom,
    ObjectProfileError,
    ObjectProfileEvaluation,
    ObjectProfileOperator,
    evaluate_object_profile,
    object_feature_catalog_data,
    object_feature_catalog_digest,
    verify_object_profile_evaluation,
)


_D = "1" * 64
_HYPOTHESIS_CATALOG_DIGEST = "2" * 64


def _binding(scenario_id: str, hypothesis_id: str) -> ObjectHypothesisBinding:
    return ObjectHypothesisBinding(
        scenario_id=scenario_id,
        hypothesis_id=hypothesis_id,
        source_component_ids=(f"component_{hypothesis_id}",),
        source_component_mask_digests=("3" * 64,),
        union_mask_digest="4" * 64,
        union_bbox=(1, 2, 11, 12),
        union_crop_digest="5" * 64,
        hypothesis_catalog_digest=_HYPOTHESIS_CATALOG_DIGEST,
    )


def _packet(
    scenario_id: str,
    values: dict[tuple[str, str], tuple[ObjectFeatureCellState, IntegerInterval | None]],
    hypothesis_ids: tuple[str, ...] = ("h0",),
) -> ObjectLocalObservationPacket:
    bindings = tuple(_binding(scenario_id, item) for item in hypothesis_ids)
    cells = []
    for hypothesis_id in hypothesis_ids:
        for feature_id in OBJECT_FEATURE_IDS:
            state, interval = values.get(
                (hypothesis_id, feature_id),
                (ObjectFeatureCellState.SCORED, IntegerInterval(0, 0)),
            )
            cells.append(
                ObjectFeatureCell(
                    hypothesis_id=hypothesis_id,
                    feature_id=feature_id,
                    state=state,
                    interval=interval,
                    reason=(
                        "observer abstained"
                        if state is ObjectFeatureCellState.INDETERMINATE
                        else "observer failed"
                        if state is ObjectFeatureCellState.ERROR
                        else None
                    ),
                    error_type=(
                        "SyntheticError"
                        if state is ObjectFeatureCellState.ERROR
                        else None
                    ),
                )
            )
    return ObjectLocalObservationPacket.create(
        scenario_id,
        bindings,
        cells,
        panel_digest=_D,
        visual_witness_packet_digest="6" * 64,
        hypothesis_catalog_digest=_HYPOTHESIS_CATALOG_DIGEST,
        feature_protocol_digest="7" * 64,
        feature_model_id="fixed-observer-v1",
        feature_receipt_digest="8" * 64,
        feature_payload_digest="9" * 64,
    )


def _profile(*atoms: ObjectProfileAtom) -> ObjectProfile:
    return ObjectProfile.create("profile-1", atoms)


def test_catalog_is_fixed_exhaustive_neutral_and_python_authoritative() -> None:
    required = {
        "straight_span_count",
        "inward_arc_count",
        "open_outline_support_ppm",
        "endpoint_count",
        "branch_count",
        "cycle_count",
        "pointed_terminal_appendage_count",
        "oblique_span_support_ppm",
        "rounded_leaf_support_ppm",
        "bird_like_support_ppm",
        "paired_sector_mismatch_support_ppm",
        "triangle_with_three_lines_support_ppm",
        "sector_like_support_ppm",
        "triangle_subshape_count",
        "additional_straight_line_count",
    }
    assert set(OBJECT_FEATURE_IDS) == required
    assert len(OBJECT_FEATURE_IDS) == len(set(OBJECT_FEATURE_IDS))
    assert object_feature_catalog_digest() == OBJECT_FEATURE_CATALOG_DIGEST
    data = object_feature_catalog_data()
    assert data["exhaustive"] is True
    assert data["python_is_canonical_authority"] is True
    assert data["lean_present"] is False
    assert data["lean_required"] is False
    assert data["lean_removal_changes_decision"] is False
    assert all(item.operational_description for item in OBJECT_FEATURE_CATALOG)
    assert all(
        item.allowed_comparators == (ObjectProfileOperator.AT_LEAST,)
        for item in OBJECT_FEATURE_CATALOG
        if item.unit == "ppm"
    )


def test_profile_grammar_is_only_positive_ordered_conjunction() -> None:
    atom = ObjectProfileAtom(
        "straight_span_count", ObjectProfileOperator.EQUALS, 2
    )
    profile = _profile(atom)
    assert ObjectProfile.from_data(profile.to_data()) == profile
    assert profile.to_data()["formula"] == "all_atoms_on_one_hypothesis"

    with pytest.raises((ObjectProfileError, TypeError), match="at least 1"):
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.EQUALS, 0)
    with pytest.raises(ObjectProfileError, match="not allowed"):
        ObjectProfileAtom(
            "bird_like_support_ppm", ObjectProfileOperator.EQUALS, 500_000
        )
    polluted = deepcopy(profile.to_data())
    polluted["polarity"] = "negative"
    with pytest.raises(ObjectProfileError, match="fields differ"):
        ObjectProfile.from_data(polluted)


def test_packet_requires_exact_exhaustive_order_and_round_trips() -> None:
    packet = _packet("s0", {})
    assert ObjectLocalObservationPacket.from_data(packet.to_data()) == packet
    assert len(packet.cells) == len(OBJECT_FEATURE_IDS)
    assert tuple(item.feature_id for item in packet.cells) == OBJECT_FEATURE_IDS

    missing = deepcopy(packet.to_data())
    missing["cells"].pop()
    with pytest.raises(ObjectProfileError, match="exhaustive"):
        ObjectLocalObservationPacket.from_data(missing)

    swapped = deepcopy(packet.to_data())
    swapped["cells"][0], swapped["cells"][1] = (
        swapped["cells"][1],
        swapped["cells"][0],
    )
    with pytest.raises(ObjectProfileError, match="exact"):
        ObjectLocalObservationPacket.from_data(swapped)


@pytest.mark.parametrize(
    ("operator", "interval", "expected"),
    (
        (ObjectProfileOperator.EQUALS, IntegerInterval(2, 2), Disposition.PRESENT),
        (ObjectProfileOperator.EQUALS, IntegerInterval(0, 1), Disposition.CERTIFIED_ABSENT),
        (ObjectProfileOperator.EQUALS, IntegerInterval(1, 3), Disposition.INDETERMINATE),
        (ObjectProfileOperator.AT_LEAST, IntegerInterval(2, 4), Disposition.PRESENT),
        (ObjectProfileOperator.AT_LEAST, IntegerInterval(0, 1), Disposition.CERTIFIED_ABSENT),
        (ObjectProfileOperator.AT_LEAST, IntegerInterval(1, 2), Disposition.INDETERMINATE),
    ),
)
def test_closed_integer_interval_semantics(
    operator: ObjectProfileOperator,
    interval: IntegerInterval,
    expected: Disposition,
) -> None:
    profile = _profile(ObjectProfileAtom("straight_span_count", operator, 2))
    packet = _packet(
        "s0",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.SCORED, interval)},
    )
    result = evaluate_object_profile(profile, (packet,))
    assert result.disposition is expected
    assert result.scenarios[0].hypotheses[0].atoms[0].disposition is expected


def test_indeterminate_abstention_is_never_promoted_by_an_interval() -> None:
    with pytest.raises(ObjectProfileError, match="no interval"):
        ObjectFeatureCell(
            "h0",
            "straight_span_count",
            ObjectFeatureCellState.INDETERMINATE,
            IntegerInterval(2, 2),
            "observer abstained",
        )
    profile = _profile(
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.AT_LEAST, 1)
    )
    packet = _packet(
        "s0",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.INDETERMINATE, None)},
    )
    assert evaluate_object_profile(profile, (packet,)).disposition is (
        Disposition.INDETERMINATE
    )


def test_conjunction_and_existential_use_proof_dominance() -> None:
    profile = _profile(
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.EQUALS, 2),
        ObjectProfileAtom("endpoint_count", ObjectProfileOperator.AT_LEAST, 1),
    )
    packet = _packet(
        "s0",
        {
            ("h0", "straight_span_count"): (
                ObjectFeatureCellState.SCORED,
                IntegerInterval(0, 0),
            ),
            ("h0", "endpoint_count"): (ObjectFeatureCellState.ERROR, None),
            ("h1", "straight_span_count"): (
                ObjectFeatureCellState.SCORED,
                IntegerInterval(2, 2),
            ),
            ("h1", "endpoint_count"): (
                ObjectFeatureCellState.SCORED,
                IntegerInterval(1, 3),
            ),
        },
        ("h0", "h1"),
    )
    result = evaluate_object_profile(profile, (packet,))
    assert result.scenarios[0].hypotheses[0].disposition is (
        Disposition.CERTIFIED_ABSENT
    )
    assert result.scenarios[0].hypotheses[1].disposition is Disposition.PRESENT
    assert result.disposition is Disposition.PRESENT


def test_empty_hypothesis_set_certifies_scenario_absence() -> None:
    profile = _profile(
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.AT_LEAST, 1)
    )
    packet = _packet("s0", {}, ())
    assert packet.hypotheses == packet.cells == ()
    assert evaluate_object_profile(profile, (packet,)).disposition is (
        Disposition.CERTIFIED_ABSENT
    )


def test_scenarios_require_unanimity_and_errors_take_precedence() -> None:
    profile = _profile(
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.AT_LEAST, 2)
    )
    present = _packet(
        "s0",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.SCORED, IntegerInterval(2, 3))},
    )
    absent = _packet(
        "s1",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.SCORED, IntegerInterval(0, 1))},
    )
    failed = _packet(
        "s2",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.ERROR, None)},
    )
    assert evaluate_object_profile(profile, (present, absent)).disposition is (
        Disposition.INDETERMINATE
    )
    assert evaluate_object_profile(profile, (present, failed)).disposition is (
        Disposition.ERROR
    )


def test_exact_digest_bindings_reject_tamper_and_cold_replay() -> None:
    profile = _profile(
        ObjectProfileAtom("straight_span_count", ObjectProfileOperator.AT_LEAST, 1)
    )
    packet = _packet(
        "s0",
        {("h0", "straight_span_count"): (ObjectFeatureCellState.SCORED, IntegerInterval(1, 2))},
    )
    result = evaluate_object_profile(profile, (packet,))
    assert ObjectProfileEvaluation.from_data(result.to_data()) == result
    assert verify_object_profile_evaluation(
        result, profile=profile, packets=(packet,)
    ) == result

    profile_tamper = deepcopy(profile.to_data())
    profile_tamper["atoms"][0]["target"] = 2
    with pytest.raises(ObjectProfileError, match="digest"):
        ObjectProfile.from_data(profile_tamper)

    packet_tamper = deepcopy(packet.to_data())
    packet_tamper["feature_payload_digest"] = "a" * 64
    with pytest.raises(ObjectProfileError, match="packet digest"):
        ObjectLocalObservationPacket.from_data(packet_tamper)

    result_tamper = deepcopy(result.to_data())
    result_tamper["lean_present"] = True
    with pytest.raises(ObjectProfileError, match="authority"):
        ObjectProfileEvaluation.from_data(result_tamper)
