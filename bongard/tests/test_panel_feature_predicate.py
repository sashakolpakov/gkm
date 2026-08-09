from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bongard.evidence import Disposition
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringQueryDecision,
    EngineeringQueryOutcome,
    EngineeringSupportTable,
    FeatureQueryDecision,
    FeatureQueryOutcome,
    FeatureSupportGapKind,
    FeatureSupportTable,
    FeatureVersionSpace,
    FeatureVocabulary,
    FrozenEngineeringFeaturePredicatePair,
    FrozenFeaturePredicatePair,
    PanelFeaturePredicateError,
    enumerate_all_of,
    evaluate_all_of,
)
from bongard.panel_soft_ontology import (
    ClosedAggregation,
    ClosedCount,
    ComponentCountParameters,
    CornerAngleClass,
    CornerAngleParameters,
    FeatureFamily,
    GestaltKind,
    GestaltResemblanceParameters,
    NativeOrientation,
    OrientationClass,
    PanelFeatureSpec,
    ReferenceFrame,
    SegmentOrientationParameters,
    SubjectScope,
)


def _d(value: str) -> str:
    return value * 64


def _specs() -> tuple[PanelFeatureSpec, ...]:
    return (
        PanelFeatureSpec(
            family=FeatureFamily.COMPONENT_COUNT,
            subject_scope=SubjectScope.WHOLE_PANEL,
            reference_frame=ReferenceFrame.NONE,
            parameters=ComponentCountParameters(ClosedCount.TWO),
        ),
        PanelFeatureSpec(
            family=FeatureFamily.GESTALT_RESEMBLANCE,
            subject_scope=SubjectScope.ONE_COHERENT_FIGURE,
            reference_frame=ReferenceFrame.NONE,
            parameters=GestaltResemblanceParameters(GestaltKind.BIRD_LIKE),
        ),
        PanelFeatureSpec(
            family=FeatureFamily.SEGMENT_ORIENTATION,
            subject_scope=SubjectScope.ONE_TRACE,
            reference_frame=ReferenceFrame.CANVAS_AXES,
            parameters=SegmentOrientationParameters(
                OrientationClass.OBLIQUE_ASCENDING,
                ClosedAggregation.ONE_WITNESSED,
            ),
        ),
        PanelFeatureSpec(
            family=FeatureFamily.CORNER_ANGLE,
            subject_scope=SubjectScope.ONE_TRACE,
            reference_frame=ReferenceFrame.LOCAL_TANGENT,
            parameters=CornerAngleParameters(
                CornerAngleClass.OBTUSE,
                ClosedAggregation.ONE_WITNESSED,
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


def _scientific_values(
    vocabulary: FeatureVocabulary,
    *,
    side0_atom: tuple[Disposition, Disposition] = (
        Disposition.PRESENT,
        Disposition.CERTIFIED_ABSENT,
    ),
    side1_atom: tuple[Disposition, Disposition] = (
        Disposition.CERTIFIED_ABSENT,
        Disposition.PRESENT,
    ),
) -> dict[tuple[str, str], Disposition]:
    side0_digest = vocabulary.side0_native_spec_digests[0]
    side1_digest = vocabulary.side1_native_spec_digests[0]
    values: dict[tuple[str, str], Disposition] = {}
    for panel in SIDE0 + SIDE1:
        physical_side = 0 if panel in SIDE0 else 1
        for spec in vocabulary.specs:
            state = Disposition.INDETERMINATE
            if spec.spec_digest == side0_digest:
                state = side0_atom[physical_side]
            elif spec.spec_digest == side1_digest:
                state = side1_atom[physical_side]
            values[(panel, spec.spec_digest)] = state
    return values


def _scientific_table(**kwargs: object) -> FeatureSupportTable:
    vocabulary = _vocabulary()
    return FeatureSupportTable.create(
        vocabulary,
        SIDE0 + SIDE1,
        _scientific_values(vocabulary, **kwargs),
    )


def _scientific_pair() -> FrozenFeaturePredicatePair:
    table = _scientific_table()
    side0 = FeatureVersionSpace.create(
        table, NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    side1 = FeatureVersionSpace.create(
        table, NativeOrientation.SIDE1_POSITIVE, SIDE0, SIDE1
    )
    return FrozenFeaturePredicatePair.create(side0, side1)


def _engineering_table() -> EngineeringSupportTable:
    vocabulary = _vocabulary()
    scientific = _scientific_values(vocabulary)
    projected = {
        key: {
            Disposition.PRESENT: EngineeringDisposition.MATCH,
            Disposition.CERTIFIED_ABSENT: EngineeringDisposition.NONMATCH,
            Disposition.INDETERMINATE: EngineeringDisposition.INDETERMINATE,
            Disposition.ERROR: EngineeringDisposition.ERROR,
        }[value]
        for key, value in scientific.items()
    }
    return EngineeringSupportTable.create(vocabulary, SIDE0 + SIDE1, projected)


def _engineering_pair() -> FrozenEngineeringFeaturePredicatePair:
    table = _engineering_table()
    side0 = EngineeringFeatureVersionSpace.create(
        table, NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    side1 = EngineeringFeatureVersionSpace.create(
        table, NativeOrientation.SIDE1_POSITIVE, SIDE0, SIDE1
    )
    return FrozenEngineeringFeaturePredicatePair.create(side0, side1)


def _query_values(
    vocabulary: FeatureVocabulary,
    side0: object,
    side1: object,
    default: object,
    panel: str,
) -> dict[tuple[str, str], object]:
    values = {(panel, spec.spec_digest): default for spec in vocabulary.specs}
    values[(panel, vocabulary.side0_native_spec_digests[0])] = side0
    values[(panel, vocabulary.side1_native_spec_digests[0])] = side1
    return values


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


def test_scientific_table_is_exact_complete_and_never_accepts_engineering_state() -> None:
    vocabulary = _vocabulary()
    values = _scientific_values(vocabulary)
    table = FeatureSupportTable.create(vocabulary, tuple(reversed(SIDE0 + SIDE1)), values)
    assert table.panel_digests == tuple(sorted(SIDE0 + SIDE1))
    assert FeatureSupportTable.from_data(table.to_data()) == table
    key = (SIDE0[0], vocabulary.side0_native_spec_digests[0])
    assert table.disposition(*key) is Disposition.PRESENT

    missing = dict(values)
    missing.pop(key)
    with pytest.raises(PanelFeaturePredicateError):
        FeatureSupportTable.create(vocabulary, SIDE0 + SIDE1, missing)
    wrong = dict(values)
    wrong[key] = EngineeringDisposition.MATCH
    with pytest.raises(TypeError):
        FeatureSupportTable.create(vocabulary, SIDE0 + SIDE1, wrong)  # type: ignore[arg-type]


def test_all_of_is_closed_positive_one_or_two_atoms_with_exact_support() -> None:
    table = _scientific_table()
    vocabulary = table.vocabulary
    formulas = enumerate_all_of(vocabulary, NativeOrientation.SIDE0_POSITIVE)
    assert [len(item.spec_digests) for item in formulas] == [1, 1, 2]
    assert all(item.to_data()["operator"] == "all_of" for item in formulas)
    assert all(item.to_data()["negation_allowed"] is False for item in formulas)
    assert evaluate_all_of(formulas[0], table, SIDE0[0]) is Disposition.PRESENT
    assert AllOf.from_data(formulas[0].to_data()) == formulas[0]

    with pytest.raises(PanelFeaturePredicateError):
        AllOf.create(vocabulary, NativeOrientation.SIDE0_POSITIVE, ())
    with pytest.raises(PanelFeaturePredicateError):
        AllOf.create(
            vocabulary,
            NativeOrientation.SIDE0_POSITIVE,
            (vocabulary.side1_native_spec_digests[0],),
        )
    injected = deepcopy(formulas[0].to_data())
    injected["not"] = True
    with pytest.raises(PanelFeaturePredicateError):
        AllOf.from_data(injected)


@pytest.mark.parametrize(
    ("native_state", "contrast_state", "expected_kind"),
    [
        (
            Disposition.INDETERMINATE,
            Disposition.CERTIFIED_ABSENT,
            FeatureSupportGapKind.NATIVE_MISS,
        ),
        (
            Disposition.PRESENT,
            Disposition.INDETERMINATE,
            FeatureSupportGapKind.UNCERTIFIED_CONTRAST,
        ),
        (
            Disposition.ERROR,
            Disposition.CERTIFIED_ABSENT,
            FeatureSupportGapKind.ERROR,
        ),
        (
            Disposition.PRESENT,
            Disposition.PRESENT,
            FeatureSupportGapKind.NO_SEPARATOR,
        ),
    ],
)
def test_empty_scientific_space_has_typed_non_laundering_gap(
    native_state: Disposition,
    contrast_state: Disposition,
    expected_kind: FeatureSupportGapKind,
) -> None:
    table = _scientific_table(side0_atom=(native_state, contrast_state))
    space = FeatureVersionSpace.create(
        table, NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    assert not space.survivor_formula_digests
    assert space.gap is not None
    assert expected_kind in space.gap.kinds
    assert FeatureVersionSpace.from_data(space.to_data()) == space


def test_exact_six_by_six_survivors_freeze_deterministically_and_roundtrip() -> None:
    table = _scientific_table()
    side0 = FeatureVersionSpace.create(
        table, NativeOrientation.SIDE0_POSITIVE, SIDE0, SIDE1
    )
    side1 = FeatureVersionSpace.create(
        table, NativeOrientation.SIDE1_POSITIVE, SIDE0, SIDE1
    )
    assert len(side0.survivor_formula_digests) == 1
    assert len(side1.survivor_formula_digests) == 1
    assert side0.gap is None and side1.gap is None
    pair = FrozenFeaturePredicatePair.create(side0, side1)
    assert len(pair.side0_predicate.formula.spec_digests) == 1
    assert FrozenFeaturePredicatePair.from_data(pair.to_data()) == pair

    with pytest.raises(PanelFeaturePredicateError):
        FeatureVersionSpace.create(
            table, NativeOrientation.SIDE0_POSITIVE, SIDE0[:5], SIDE1
        )
    with pytest.raises(PanelFeaturePredicateError):
        FrozenFeaturePredicatePair.create(side1, side0)


@pytest.mark.parametrize(
    ("side0", "side1", "outcome"),
    [
        (
            Disposition.PRESENT,
            Disposition.CERTIFIED_ABSENT,
            FeatureQueryOutcome.SIDE0,
        ),
        (
            Disposition.CERTIFIED_ABSENT,
            Disposition.PRESENT,
            FeatureQueryOutcome.SIDE1,
        ),
        (Disposition.PRESENT, Disposition.PRESENT, FeatureQueryOutcome.ABSTAIN),
        (
            Disposition.CERTIFIED_ABSENT,
            Disposition.CERTIFIED_ABSENT,
            FeatureQueryOutcome.ABSTAIN,
        ),
        (Disposition.ERROR, Disposition.CERTIFIED_ABSENT, FeatureQueryOutcome.ERROR),
    ],
)
def test_scientific_query_requires_a_two_sided_witness(
    side0: Disposition, side1: Disposition, outcome: FeatureQueryOutcome
) -> None:
    pair = _scientific_pair()
    panel = _d("0")
    vocabulary = pair.vocabulary
    table = FeatureSupportTable.create(
        vocabulary,
        (panel,),
        _query_values(
            vocabulary, side0, side1, Disposition.INDETERMINATE, panel
        ),  # type: ignore[arg-type]
    )
    decision = FeatureQueryDecision.create(pair, table, panel)
    assert decision.outcome is outcome
    assert FeatureQueryDecision.from_data(decision.to_data()) == decision
    tampered = deepcopy(decision.to_data())
    tampered["outcome"] = (
        FeatureQueryOutcome.SIDE0.value
        if outcome is not FeatureQueryOutcome.SIDE0
        else FeatureQueryOutcome.SIDE1.value
    )
    with pytest.raises(PanelFeaturePredicateError):
        FeatureQueryDecision.from_data(tampered)


def test_engineering_lane_is_distinct_labelled_and_roundtrips_end_to_end() -> None:
    table = _engineering_table()
    data = table.to_data()
    assert data["engineering_only"] is True
    assert data["uncalibrated"] is True
    assert data["scientific_evidence"] is False
    assert EngineeringSupportTable.from_data(data) == table

    scientific_values = _scientific_values(table.vocabulary)
    with pytest.raises(TypeError):
        EngineeringSupportTable.create(
            table.vocabulary, SIDE0 + SIDE1, scientific_values  # type: ignore[arg-type]
        )

    pair = _engineering_pair()
    assert FrozenEngineeringFeaturePredicatePair.from_data(pair.to_data()) == pair
    panel = _d("0")
    query = EngineeringSupportTable.create(
        pair.vocabulary,
        (panel,),
        _query_values(
            pair.vocabulary,
            EngineeringDisposition.MATCH,
            EngineeringDisposition.NONMATCH,
            EngineeringDisposition.INDETERMINATE,
            panel,
        ),  # type: ignore[arg-type]
    )
    decision = EngineeringQueryDecision.create(pair, query, panel)
    assert decision.outcome is EngineeringQueryOutcome.SIDE0
    assert EngineeringQueryDecision.from_data(decision.to_data()) == decision
    rendered = json.dumps(decision.to_data(), sort_keys=True).lower()
    assert "lean" not in rendered


def test_multivalued_artifacts_and_outcomes_refuse_boolean_coercion() -> None:
    pair = _scientific_pair()
    formula = pair.side0_predicate.formula
    table = pair.side0_predicate.version_space.support_table
    for value in (
        table.vocabulary,
        table,
        formula,
        pair.side0_predicate.version_space,
        pair.side0_predicate,
        pair,
        FeatureQueryOutcome.ABSTAIN,
        EngineeringDisposition.NONMATCH,
        EngineeringQueryOutcome.ABSTAIN,
    ):
        with pytest.raises(TypeError):
            bool(value)


def test_vnext_artifact_data_has_no_proof_assistant_fields_or_values() -> None:
    scientific = _scientific_pair().to_data()
    engineering = _engineering_pair().to_data()
    for artifact in (scientific, engineering):
        rendered = json.dumps(artifact, sort_keys=True).lower()
        assert "lean" not in rendered
        assert '"not"' not in rendered
        assert '"operator": "all_of"' in rendered
