from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringQueryDecision,
    EngineeringQueryOutcome,
    EngineeringSupportTable,
    FeatureSupportGapKind,
    FeatureSupportTable,
    FeatureVersionSpace,
    FeatureVocabulary,
    FrozenEngineeringFeaturePredicatePair,
    FrozenFeaturePredicate,
    PanelFeaturePredicateError,
    ScientificFeatureProjectionRecord,
    enumerate_all_of,
    evaluate_all_of,
    evaluate_engineering_all_of,
)
import bongard.panel_soft_ontology as o


def _d(value: str) -> str:
    return value * 64


def _specs() -> tuple[o.PanelFeatureSpec, ...]:
    return (
        o.PanelFeatureSpec(
            family=o.FeatureFamily.COMPONENT_COUNT,
            subject_scope=o.SubjectScope.WHOLE_PANEL,
            reference_frame=o.ReferenceFrame.NONE,
            parameters=o.ComponentCountParameters(o.ClosedCount.TWO),
        ),
        o.PanelFeatureSpec(
            family=o.FeatureFamily.GESTALT_RESEMBLANCE,
            subject_scope=o.SubjectScope.ONE_COHERENT_FIGURE,
            reference_frame=o.ReferenceFrame.NONE,
            parameters=o.GestaltResemblanceParameters(o.GestaltKind.BIRD_LIKE),
        ),
        o.PanelFeatureSpec(
            family=o.FeatureFamily.SEGMENT_ORIENTATION,
            subject_scope=o.SubjectScope.ONE_TRACE,
            reference_frame=o.ReferenceFrame.CANVAS_AXES,
            parameters=o.SegmentOrientationParameters(
                o.OrientationClass.OBLIQUE_ASCENDING,
                o.ClosedAggregation.ONE_WITNESSED,
            ),
        ),
        o.PanelFeatureSpec(
            family=o.FeatureFamily.CORNER_ANGLE,
            subject_scope=o.SubjectScope.ONE_TRACE,
            reference_frame=o.ReferenceFrame.LOCAL_TANGENT,
            parameters=o.CornerAngleParameters(
                o.CornerAngleClass.OBTUSE,
                o.ClosedAggregation.ONE_WITNESSED,
            ),
        ),
    )


def _vocabulary() -> FeatureVocabulary:
    first, second, third, fourth = _specs()
    return FeatureVocabulary.create(
        side0_specs=(second, first, first),
        side1_specs=(fourth, third, third),
    )


SIDE0 = tuple(_d(item) for item in "123456")
SIDE1 = tuple(_d(item) for item in "abcdef")


def _assessment(
    risk: o.CalibrationRisk, population_char: str
) -> o.CalibrationAssessment:
    return o.CalibrationAssessment(
        risk,
        _d(population_char),
        _d("3"),
        200,
        50_000,
        20_000,
        950_000,
        100,
        200,
        _d("4"),
    )


def _presence_token(spec: o.PanelFeatureSpec, index: int) -> tuple[object, object]:
    domain = o.FeatureDomain(
        spec.family,
        spec.subject_scope,
        spec.reference_frame,
        (spec,),
    )
    grant = o.PresenceCalibrationGrant(
        domain,
        _d("d"),
        _d("7"),
        _assessment(o.CalibrationRisk.FALSE_POSITIVE_CLAIM, "8"),
        _d("9"),
    )
    authority = o.FeatureCalibrationAuthority(
        f"calibration.predicate-fixture-{index}.v1",
        domain,
        _d("d"),
        _d("c"),
        _d("e"),
        grant,
    )
    token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.PRESENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=grant.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    return authority, token


def _inventory(panel: str, *, complete: bool = False) -> o.OwnerInventory:
    return o.OwnerInventory(
        panel,
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        complete,
        (),
    )


def _scientific_records(
    vocabulary: FeatureVocabulary,
    panels: tuple[str, ...],
    *,
    state: o.RawMeasurementState = o.RawMeasurementState.UNRESOLVED,
) -> tuple[
    dict[tuple[str, str], ScientificFeatureProjectionRecord],
    dict[str, ScientificFeatureProjectionRecord],
    dict[str, object],
]:
    if state not in {o.RawMeasurementState.UNRESOLVED, o.RawMeasurementState.ERROR}:
        raise AssertionError("fixture only creates nonterminal measurements")
    tokens = {
        spec.spec_digest: _presence_token(spec, index)[1]
        for index, spec in enumerate(vocabulary.specs)
    }
    projections: dict[tuple[str, str], ScientificFeatureProjectionRecord] = {}
    for panel in panels:
        inventory = _inventory(panel)
        for spec in vocabulary.specs:
            raw = o.RawFeatureMeasurement(
                spec,
                inventory,
                _d("d"),
                _d("7"),
                state,
                issue_code=(
                    o.MeasurementIssueCode.AMBIGUOUS_OWNER
                    if state is o.RawMeasurementState.UNRESOLVED
                    else o.MeasurementIssueCode.OBSERVER_FAILURE
                ),
            )
            receipt = canonical_digest(
                {
                    "schema": "test-projection-receipt.v1",
                    "panel_digest": panel,
                    "spec_digest": spec.spec_digest,
                    "measurement_digest": raw.measurement_digest,
                }
            )
            record = ScientificFeatureProjectionRecord.create(
                panel_digest=panel,
                spec=spec,
                raw_measurement=raw,
                verified_authority=tokens[spec.spec_digest],
                verified_custody=None,
                projection_receipt_digest=receipt,
            )
            projections[(panel, spec.spec_digest)] = record
    registry = {
        record.projection_record_digest: record for record in projections.values()
    }
    return projections, registry, tokens


def _scientific_table(
    *, state: o.RawMeasurementState = o.RawMeasurementState.UNRESOLVED
) -> tuple[FeatureSupportTable, dict[str, ScientificFeatureProjectionRecord]]:
    vocabulary = _vocabulary()
    projections, registry, _ = _scientific_records(
        vocabulary, SIDE0 + SIDE1, state=state
    )
    return (
        FeatureSupportTable.create(vocabulary, SIDE0 + SIDE1, projections),
        registry,
    )


def _engineering_values(
    vocabulary: FeatureVocabulary,
    panels: tuple[str, ...] = SIDE0 + SIDE1,
) -> dict[tuple[str, str], EngineeringDisposition]:
    side0_digest = vocabulary.side0_native_spec_digests[0]
    side1_digest = vocabulary.side1_native_spec_digests[0]
    values: dict[tuple[str, str], EngineeringDisposition] = {}
    for panel in panels:
        physical_side = 0 if panel in SIDE0 else 1
        for spec in vocabulary.specs:
            state = EngineeringDisposition.INDETERMINATE
            if spec.spec_digest == side0_digest:
                state = (
                    EngineeringDisposition.MATCH
                    if physical_side == 0
                    else EngineeringDisposition.NONMATCH
                )
            elif spec.spec_digest == side1_digest:
                state = (
                    EngineeringDisposition.NONMATCH
                    if physical_side == 0
                    else EngineeringDisposition.MATCH
                )
            values[(panel, spec.spec_digest)] = state
    return values


def _engineering_table() -> EngineeringSupportTable:
    vocabulary = _vocabulary()
    return EngineeringSupportTable.create(
        vocabulary, SIDE0 + SIDE1, _engineering_values(vocabulary)
    )


def _engineering_pair() -> FrozenEngineeringFeaturePredicatePair:
    table = _engineering_table()
    side0 = EngineeringFeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    side1 = EngineeringFeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE1_POSITIVE, SIDE0, SIDE1
    )
    return FrozenEngineeringFeaturePredicatePair.create(side0, side1)


def _query_values(
    vocabulary: FeatureVocabulary,
    side0: EngineeringDisposition,
    side1: EngineeringDisposition,
    panel: str,
) -> dict[tuple[str, str], EngineeringDisposition]:
    values = {
        (panel, spec.spec_digest): EngineeringDisposition.INDETERMINATE
        for spec in vocabulary.specs
    }
    values[(panel, vocabulary.side0_native_spec_digests[0])] = side0
    values[(panel, vocabulary.side1_native_spec_digests[0])] = side1
    return values


def _certified_absence_projection() -> tuple[
    ScientificFeatureProjectionRecord, object, object
]:
    panel = _d("0")
    spec = o.PanelFeatureSpec(
        o.FeatureFamily.OPEN_TRACE,
        o.SubjectScope.ONE_TRACE,
        o.ReferenceFrame.NONE,
        o.OpenTraceParameters(o.OpenTraceKind.SIMPLE_UNBRANCHED),
    )
    inventory = _inventory(panel, complete=True)
    search_domain = o.SearchResolutionDomain.for_spec(spec)
    empty = o.EmptyEligibleDomainCertificate(
        inventory.inventory_digest,
        search_domain.domain_digest,
        inventory.enumeration_receipt_digest,
        _d("6"),
    )
    absence = o.AbsenceCertificate(
        spec,
        inventory,
        _d("d"),
        search_domain,
        _d("1"),
        (),
        (),
        True,
        _d("2"),
        empty,
    )
    domain = o.FeatureDomain(
        spec.family, spec.subject_scope, spec.reference_frame, (spec,)
    )
    grant = o.AbsenceCalibrationGrant(
        domain,
        _d("d"),
        _d("7"),
        inventory.enumeration_protocol_digest,
        absence.search_protocol_digest,
        _assessment(o.CalibrationRisk.FALSE_NEGATIVE_CLAIM, "a"),
        _assessment(o.CalibrationRisk.OWNER_INVENTORY_OMISSION, "b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        (o.RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH,),
        _d("9"),
    )
    authority = o.FeatureCalibrationAuthority(
        "calibration.predicate-absence-fixture.v1",
        domain,
        _d("d"),
        _d("c"),
        _d("e"),
        absence_grant=grant,
    )
    token = o.verify_feature_calibration_authority(
        authority,
        capability=o.CalibrationCapability.ABSENCE,
        expected_authority_digest=authority.authority_digest,
        expected_grant_digest=grant.grant_digest,
        trusted_root_digest=_d("c"),
        verifier_receipt_digest=_d("f"),
        campaign_time_unix=150,
    )
    raw = o.RawFeatureMeasurement(
        spec,
        inventory,
        _d("d"),
        _d("7"),
        o.RawMeasurementState.EXHAUSTIVE_SEARCH_NEGATIVE,
        absence=absence,
    )
    custody = o.verify_raw_measurement_custody(
        raw,
        expected_measurement_digest=raw.measurement_digest,
        expected_inventory_digest=inventory.inventory_digest,
        expected_enumeration_receipt_digest=inventory.enumeration_receipt_digest,
        expected_evidence_receipt_digest=absence.search_receipt_digest,
        verifier_receipt_digest=_d("5"),
    )
    record = ScientificFeatureProjectionRecord.create(
        panel_digest=panel,
        spec=spec,
        raw_measurement=raw,
        verified_authority=token,
        verified_custody=custody,
        projection_receipt_digest=_d("4"),
    )
    return record, token, custody


def test_vocabulary_is_exactly_deduplicated_canonical_and_tamper_evident() -> None:
    vocabulary = _vocabulary()
    assert len(vocabulary.specs) == 4
    assert len(vocabulary.side0_native_spec_digests) == 2
    assert len(vocabulary.side1_native_spec_digests) == 2
    assert FeatureVocabulary.from_data(vocabulary.to_data()) == vocabulary

    reversed_data = deepcopy(vocabulary.to_data())
    reversed_data["specs"].reverse()
    with pytest.raises(PanelFeaturePredicateError):
        FeatureVocabulary.from_data(reversed_data)
    injected = deepcopy(vocabulary.to_data())
    injected["polarity"] = "reverse"
    with pytest.raises(PanelFeaturePredicateError):
        FeatureVocabulary.from_data(injected)


def test_scientific_projection_binds_raw_authority_custody_and_replays() -> None:
    record, token, custody = _certified_absence_projection()
    assert record.disposition is Disposition.CERTIFIED_ABSENT
    data = record.to_data()
    assert data["raw_measurement"]["state"] == "exhaustive_search_negative"
    assert data["calibration_authority"]["authority_id"].startswith("calibration.")
    assert data["custody_verification"]["evidence_receipt_digest"] == _d("2")
    assert data["projection_receipt_digest"] == _d("4")
    assert (
        ScientificFeatureProjectionRecord.from_data_verified(
            data, verified_authority=token, verified_custody=custody
        )
        == record
    )

    tampered = deepcopy(data)
    tampered["projected_disposition"] = Disposition.PRESENT.value
    with pytest.raises(PanelFeaturePredicateError):
        ScientificFeatureProjectionRecord.from_data_verified(
            tampered, verified_authority=token, verified_custody=custody
        )


def test_scientific_table_rejects_naked_dispositions_and_requires_reverified_records() -> None:
    vocabulary = _vocabulary()
    projections, registry, _ = _scientific_records(vocabulary, SIDE0 + SIDE1)
    table = FeatureSupportTable.create(
        vocabulary, tuple(reversed(SIDE0 + SIDE1)), projections
    )
    assert table.panel_digests == tuple(sorted(SIDE0 + SIDE1))
    assert (
        FeatureSupportTable.from_data(
            table.to_data(), verified_projection_records=registry
        )
        == table
    )
    key = (SIDE0[0], vocabulary.side0_native_spec_digests[0])
    assert table.disposition(*key) is Disposition.INDETERMINATE

    naked = {key_: Disposition.CERTIFIED_ABSENT for key_ in projections}
    with pytest.raises(TypeError):
        FeatureSupportTable.create(
            vocabulary, SIDE0 + SIDE1, naked  # type: ignore[arg-type]
        )
    missing = dict(projections)
    missing.pop(key)
    with pytest.raises(PanelFeaturePredicateError):
        FeatureSupportTable.create(vocabulary, SIDE0 + SIDE1, missing)
    with pytest.raises(TypeError):
        FeatureSupportTable.from_data(table.to_data())  # type: ignore[call-arg]


def test_all_of_is_closed_positive_and_genuine_missing_evidence_stays_indeterminate() -> None:
    table, _ = _scientific_table()
    vocabulary = table.vocabulary
    formulas = enumerate_all_of(vocabulary, o.NativeOrientation.SIDE0_POSITIVE)
    assert [len(item.spec_digests) for item in formulas] == [1, 1, 2]
    assert all(item.to_data()["operator"] == "all_of" for item in formulas)
    assert all(item.to_data()["negation_allowed"] is False for item in formulas)
    assert evaluate_all_of(formulas[0], table, SIDE0[0]) is Disposition.INDETERMINATE
    assert AllOf.from_data(formulas[0].to_data()) == formulas[0]

    with pytest.raises(PanelFeaturePredicateError):
        AllOf.create(vocabulary, o.NativeOrientation.SIDE0_POSITIVE, ())
    with pytest.raises(PanelFeaturePredicateError):
        AllOf.create(
            vocabulary,
            o.NativeOrientation.SIDE0_POSITIVE,
            (vocabulary.side1_native_spec_digests[0],),
        )
    injected = deepcopy(formulas[0].to_data())
    injected["not"] = True
    with pytest.raises(PanelFeaturePredicateError):
        AllOf.from_data(injected)


def test_python_predicate_treats_straight_count_as_a_distinct_positive_atom() -> None:
    straight = o.PanelFeatureSpec(
        o.FeatureFamily.STRAIGHT_SEGMENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.StraightSegmentCountParameters(o.ClosedCount.TWO),
    )
    generic = o.PanelFeatureSpec(
        o.FeatureFamily.EXACT_SEGMENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ExactSegmentCountParameters(o.ClosedCount.TWO),
    )
    assert straight.spec_digest != generic.spec_digest
    vocabulary = FeatureVocabulary.create(
        side0_specs=(straight,), side1_specs=(generic,)
    )
    panel = _d("0")
    table = EngineeringSupportTable.create(
        vocabulary,
        (panel,),
        {
            (panel, straight.spec_digest): EngineeringDisposition.MATCH,
            (panel, generic.spec_digest): EngineeringDisposition.NONMATCH,
        },
    )
    formula = AllOf.create(
        vocabulary,
        o.NativeOrientation.SIDE0_POSITIVE,
        (straight.spec_digest,),
    )
    assert (
        evaluate_engineering_all_of(formula, table, panel)
        is EngineeringDisposition.MATCH
    )
    assert "lean" not in json.dumps(formula.to_data(), sort_keys=True).lower()


@pytest.mark.parametrize(
    ("state", "expected_kind"),
    [
        (o.RawMeasurementState.UNRESOLVED, FeatureSupportGapKind.NATIVE_MISS),
        (o.RawMeasurementState.ERROR, FeatureSupportGapKind.ERROR),
    ],
)
def test_verified_missing_or_error_evidence_yields_gap_not_survivor(
    state: o.RawMeasurementState, expected_kind: FeatureSupportGapKind
) -> None:
    table, registry = _scientific_table(state=state)
    space = FeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    assert not space.survivor_formula_digests
    assert space.gap is not None and expected_kind in space.gap.kinds
    assert (
        FeatureVersionSpace.from_data(
            space.to_data(), verified_projection_records=registry
        )
        == space
    )
    with pytest.raises(PanelFeaturePredicateError):
        FrozenFeaturePredicate.create(space)


def test_scientific_support_still_requires_exact_six_by_six() -> None:
    table, _ = _scientific_table()
    with pytest.raises(PanelFeaturePredicateError):
        FeatureVersionSpace.create(
            table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0[:5], SIDE1
        )
    with pytest.raises(PanelFeaturePredicateError):
        FeatureVersionSpace.create(
            table,
            o.NativeOrientation.SIDE0_POSITIVE,
            SIDE0,
            SIDE1[:-1] + (SIDE0[0],),
        )


def test_engineering_survivors_freeze_deterministically_and_roundtrip() -> None:
    table = _engineering_table()
    side0 = EngineeringFeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    side1 = EngineeringFeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE1_POSITIVE, SIDE0, SIDE1
    )
    assert len(side0.survivor_formula_digests) == 1
    assert len(side1.survivor_formula_digests) == 1
    pair = FrozenEngineeringFeaturePredicatePair.create(side0, side1)
    assert len(pair.side0_predicate.formula.spec_digests) == 1
    assert FrozenEngineeringFeaturePredicatePair.from_data(pair.to_data()) == pair
    with pytest.raises(PanelFeaturePredicateError):
        FrozenEngineeringFeaturePredicatePair.create(side1, side0)


@pytest.mark.parametrize(
    ("side0", "side1", "outcome"),
    [
        (
            EngineeringDisposition.MATCH,
            EngineeringDisposition.NONMATCH,
            EngineeringQueryOutcome.SIDE0,
        ),
        (
            EngineeringDisposition.NONMATCH,
            EngineeringDisposition.MATCH,
            EngineeringQueryOutcome.SIDE1,
        ),
        (
            EngineeringDisposition.MATCH,
            EngineeringDisposition.MATCH,
            EngineeringQueryOutcome.ABSTAIN,
        ),
        (
            EngineeringDisposition.NONMATCH,
            EngineeringDisposition.NONMATCH,
            EngineeringQueryOutcome.ABSTAIN,
        ),
        (
            EngineeringDisposition.ERROR,
            EngineeringDisposition.NONMATCH,
            EngineeringQueryOutcome.ERROR,
        ),
    ],
)
def test_engineering_query_requires_a_two_sided_witness(
    side0: EngineeringDisposition,
    side1: EngineeringDisposition,
    outcome: EngineeringQueryOutcome,
) -> None:
    pair = _engineering_pair()
    panel = _d("0")
    query = EngineeringSupportTable.create(
        pair.vocabulary,
        (panel,),
        _query_values(pair.vocabulary, side0, side1, panel),
    )
    decision = EngineeringQueryDecision.create(pair, query, panel)
    assert decision.outcome is outcome
    assert EngineeringQueryDecision.from_data(decision.to_data()) == decision
    tampered = deepcopy(decision.to_data())
    tampered["outcome"] = (
        EngineeringQueryOutcome.SIDE0.value
        if outcome is not EngineeringQueryOutcome.SIDE0
        else EngineeringQueryOutcome.SIDE1.value
    )
    with pytest.raises(PanelFeaturePredicateError):
        EngineeringQueryDecision.from_data(tampered)


def test_engineering_lane_is_distinct_labelled_and_never_scientific_input() -> None:
    table = _engineering_table()
    data = table.to_data()
    assert data["engineering_only"] is True
    assert data["uncalibrated"] is True
    assert data["scientific_evidence"] is False
    assert EngineeringSupportTable.from_data(data) == table

    scientific, _, _ = _scientific_records(table.vocabulary, SIDE0 + SIDE1)
    with pytest.raises(TypeError):
        EngineeringSupportTable.create(
            table.vocabulary, SIDE0 + SIDE1, scientific  # type: ignore[arg-type]
        )


def test_multivalued_artifacts_and_outcomes_refuse_boolean_coercion() -> None:
    table, _ = _scientific_table()
    space = FeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    pair = _engineering_pair()
    formula = pair.side0_predicate.formula
    for value in (
        table.vocabulary,
        table.cells[0].projection,
        table,
        space,
        formula,
        pair.side0_predicate,
        pair,
        EngineeringDisposition.NONMATCH,
        EngineeringQueryOutcome.ABSTAIN,
    ):
        with pytest.raises(TypeError):
            bool(value)


def test_vnext_artifact_data_has_no_proof_assistant_or_negation_fields() -> None:
    table, _ = _scientific_table()
    scientific = FeatureVersionSpace.create(
        table, o.NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    ).to_data()
    engineering = _engineering_pair().to_data()
    for artifact in (scientific, engineering):
        rendered = json.dumps(artifact, sort_keys=True).lower()
        assert "lean" not in rendered
        assert '"not"' not in rendered
        assert '"operator": "all_of"' in rendered
