from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import Atom, Relation, StaticLegCall, evaluate_formula, validate_formula
from bongard.legs import LegRegistry, TypedValue
from bongard.support_prototypes import (
    INPUT_CONTRACT,
    ORIENTATION,
    ContrastiveMargin,
    FeatureDimension,
    FeatureInterval,
    FrozenFeatureSpace,
    FrozenPanelFeatures,
    FrozenSupportPrototypes,
    PositivePrototypeFormula,
    SUPPORT_PROTOTYPE_FEATURES,
    SupportPrototypeError,
    SupportPrototypeIntegrityError,
    SupportPrototypePlan,
    contrastive_margin,
    evaluate_support_prototype,
    fit_support_prototypes,
    panel_side_assignment_digest,
    register_support_prototype_leg,
    validate_prototype_formula,
    verify_support_prototypes,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_space() -> FrozenFeatureSpace:
    return FrozenFeatureSpace(
        extractor_id="universal-shape-features",
        extractor_version="1",
        extractor_artifact_digest=digest("extractor"),
        preprocessing_digest=digest("preprocessing"),
        receipt_protocol_digest=digest("receipt protocol"),
        dimensions=(
            FeatureDimension("angle_obliqueness", "fraction", 0.0, 1.0, 1.0),
            FeatureDimension("bilateral_balance", "fraction", 0.0, 1.0, 1.0),
        ),
    )


def vector(
    name: str,
    angle: float | tuple[float, float],
    balance: float | tuple[float, float],
    *,
    space: FrozenFeatureSpace | None = None,
) -> FrozenPanelFeatures:
    frozen_space = space or feature_space()

    def interval(feature: str, value: float | tuple[float, float]) -> FeatureInterval:
        lower, upper = value if isinstance(value, tuple) else (value, value)
        return FeatureInterval(feature, lower, upper)

    return FrozenPanelFeatures(
        panel_digest=digest("panel:" + name),
        feature_space_digest=frozen_space.digest(),
        extractor_receipt_digest=digest("receipt:" + name),
        values=(
            interval("angle_obliqueness", angle),
            interval("bilateral_balance", balance),
        ),
    )


def support() -> tuple[
    FrozenFeatureSpace,
    SupportPrototypePlan,
    tuple[FrozenPanelFeatures, ...],
    tuple[FrozenPanelFeatures, ...],
    FrozenSupportPrototypes,
]:
    space = feature_space()
    positive = (
        vector("positive-a", 0.8, 0.9, space=space),
        vector("positive-b", 0.9, 0.8, space=space),
    )
    negative = (
        vector("negative-a", 0.1, 0.2, space=space),
        vector("negative-b", 0.2, 0.1, space=space),
    )
    plan = SupportPrototypePlan(
        space.digest(),
        panel_side_assignment_digest(
            tuple(item.panel_digest for item in positive),
            tuple(item.panel_digest for item in negative),
        ),
        2,
    )
    artifact = fit_support_prototypes(
        plan,
        space,
        positive,
        negative,
        expected_plan_digest=plan.digest(),
    )
    return space, plan, positive, negative, artifact


def formula(
    space: FrozenFeatureSpace, artifact: FrozenSupportPrototypes
) -> PositivePrototypeFormula:
    return PositivePrototypeFormula(
        claim="the shape has the positive support's balanced oblique geometry",
        feature_space_digest=space.digest(),
        prototype_digest=artifact.digest(),
        support_assignment_digest=artifact.support_assignment_digest,
        decision_margin=0.1,
    )


def origin(name: str) -> Provenance:
    return Provenance(
        producer="test-feature-extractor",
        version="1",
        method=name,
        input_digests=(digest(name),),
    )


def test_feature_packet_is_candidate_independent_by_schema() -> None:
    packet = vector("one", 0.5, 0.6)
    assert set(packet.to_data()) == {
        "schema",
        "panel_digest",
        "feature_space_digest",
        "extractor_receipt_digest",
        "values",
    }
    forbidden = {
        "task_id",
        "side",
        "positive",
        "query_role",
        "claim",
        "formula",
        "action_program",
    }
    assert forbidden.isdisjoint(packet.to_data())
    assert feature_space().to_data()["input_contract"] == INPUT_CONTRACT


def test_feature_space_and_packet_have_strict_backend_neutral_round_trips() -> None:
    space = feature_space()
    packet = vector("round-trip", (0.2, 0.3), (0.6, 0.7), space=space)
    space_data = json.loads(json.dumps(space.to_data()))
    packet_data = json.loads(json.dumps(packet.to_data()))
    assert FrozenFeatureSpace.from_data(space_data) == space
    assert FrozenPanelFeatures.from_data(packet_data) == packet
    assert FrozenFeatureSpace.from_data(space_data).digest() == space.digest()
    assert FrozenPanelFeatures.from_data(packet_data).digest() == packet.digest()

    space_data["input_contract"] = "panel_plus_candidate"
    with pytest.raises(ValueError, match="task-relative"):
        FrozenFeatureSpace.from_data(space_data)


def test_feature_space_rejects_ambiguous_coordinates_and_scaling() -> None:
    with pytest.raises(ValueError, match="name-sorted"):
        replace(feature_space(), dimensions=tuple(reversed(feature_space().dimensions)))
    with pytest.raises(ValueError, match="positive"):
        FeatureDimension("bad_scale", "fraction", 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="below"):
        FeatureDimension("bad_bounds", "fraction", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        FeatureInterval("bad_value", 0.0, float("inf"))


def test_feature_packet_validation_binds_space_coordinates_and_bounds() -> None:
    space = feature_space()
    vector("valid", 0.2, 0.8, space=space).validate(space)
    wrong_space = replace(space, extractor_version="2")
    with pytest.raises(SupportPrototypeIntegrityError, match="another feature space"):
        vector("valid", 0.2, 0.8, space=space).validate(wrong_space)
    outside = vector("outside", -0.1, 0.8, space=space)
    with pytest.raises(SupportPrototypeIntegrityError, match="outside"):
        outside.validate(space)


def test_vector_identity_binds_panel_receipt_and_every_interval() -> None:
    packet = vector("identity", 0.2, 0.8)
    changed_value = replace(
        packet,
        values=(FeatureInterval("angle_obliqueness", 0.3, 0.3), packet.values[1]),
    )
    assert len({packet.digest(), changed_value.digest()}) == 2
    assert packet.digest() != replace(packet, panel_digest=digest("other panel")).digest()
    assert packet.digest() != replace(
        packet, extractor_receipt_digest=digest("other receipt")
    ).digest()


def test_fit_is_support_only_order_invariant_and_replayable() -> None:
    space, plan, positive, negative, artifact = support()
    reordered = fit_support_prototypes(
        plan,
        space,
        tuple(reversed(positive)),
        tuple(reversed(negative)),
        expected_plan_digest=plan.digest(),
    )
    assert reordered == artifact
    assert [item.lower for item in artifact.positive_centroid] == pytest.approx(
        [0.85, 0.85]
    )
    assert [item.upper for item in artifact.positive_centroid] == pytest.approx(
        [0.85, 0.85]
    )
    assert [item.lower for item in artifact.negative_centroid] == pytest.approx(
        [0.15, 0.15]
    )
    assert [item.upper for item in artifact.negative_centroid] == pytest.approx(
        [0.15, 0.15]
    )
    verify_support_prototypes(artifact, plan, space, positive, negative)


def test_fit_rejects_unfrozen_plan_small_sides_and_duplicate_panels() -> None:
    space, plan, positive, negative, _ = support()
    with pytest.raises(SupportPrototypeIntegrityError, match="frozen commitment"):
        fit_support_prototypes(
            plan,
            space,
            positive,
            negative,
            expected_plan_digest=digest("wrong plan"),
        )
    with pytest.raises(SupportPrototypeError, match="insufficient"):
        fit_support_prototypes(
            plan,
            space,
            positive[:1],
            negative,
            expected_plan_digest=plan.digest(),
        )


def test_frozen_panel_side_commitment_rejects_post_hoc_polarity_swap() -> None:
    space, plan, positive, negative, _ = support()
    assert panel_side_assignment_digest(
        tuple(item.panel_digest for item in positive),
        tuple(item.panel_digest for item in negative),
    ) != panel_side_assignment_digest(
        tuple(item.panel_digest for item in negative),
        tuple(item.panel_digest for item in positive),
    )
    with pytest.raises(SupportPrototypeIntegrityError, match="support sides differ"):
        fit_support_prototypes(
            plan,
            space,
            negative,
            positive,
            expected_plan_digest=plan.digest(),
        )
    duplicate = replace(negative[0], panel_digest=positive[0].panel_digest)
    with pytest.raises(SupportPrototypeError, match="unique"):
        fit_support_prototypes(
            plan,
            space,
            positive,
            (duplicate, negative[1]),
            expected_plan_digest=plan.digest(),
        )


def test_prototype_artifact_binds_exact_side_membership_and_preimage() -> None:
    space, plan, positive, negative, artifact = support()
    data = json.loads(json.dumps(artifact.to_data()))
    assert FrozenSupportPrototypes.from_data(data) == artifact
    assert "polarity" not in data
    assert data["orientation"] == ORIENTATION
    assert not ({"task_id", "action_program"} & set(data))

    altered = replace(
        artifact,
        positive_centroid=(
            FeatureInterval("angle_obliqueness", 0.8, 0.8),
            artifact.positive_centroid[1],
        ),
    )
    with pytest.raises(SupportPrototypeIntegrityError, match="preimage"):
        verify_support_prototypes(altered, plan, space, positive, negative)


def test_formula_identity_binds_claim_margin_space_and_prototype() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    data = json.loads(json.dumps(predicate.to_data()))
    assert PositivePrototypeFormula.from_data(data) == predicate
    assert data["orientation"] == ORIENTATION
    assert "polarity" not in data
    assert predicate.digest() != replace(predicate, claim="another claim").digest()
    assert predicate.digest() != replace(predicate, decision_margin=0.2).digest()
    assert predicate.digest() != replace(
        predicate, prototype_digest=digest("other prototype")
    ).digest()
    with pytest.raises(ValueError, match="strictly positive"):
        replace(predicate, decision_margin=0.0)


def test_decoders_reject_polarity_flip_and_unknown_fields() -> None:
    space, plan, _, _, artifact = support()
    predicate = formula(space, artifact)
    plan_data = plan.to_data()
    plan_data["orientation"] = "positive_distance_minus_negative_distance"
    with pytest.raises(ValueError, match="polarity"):
        SupportPrototypePlan.from_data(plan_data)
    formula_data = predicate.to_data()
    formula_data["orientation"] = "positive_distance_minus_negative_distance"
    with pytest.raises(ValueError, match="polarity"):
        PositivePrototypeFormula.from_data(formula_data)
    artifact_data = artifact.to_data()
    artifact_data["negate"] = True
    with pytest.raises(ValueError, match="fields differ"):
        FrozenSupportPrototypes.from_data(artifact_data)


def test_fixed_contrastive_margin_points_toward_positive_support() -> None:
    space, _, _, _, artifact = support()
    positive_query = vector("query-positive", 0.84, 0.86, space=space)
    negative_query = vector("query-negative", 0.14, 0.16, space=space)
    positive_score = contrastive_margin(positive_query, artifact, space)
    negative_score = contrastive_margin(negative_query, artifact, space)
    assert positive_score.lower == pytest.approx(0.69)
    assert positive_score.upper == pytest.approx(0.69)
    assert negative_score.lower == pytest.approx(-0.69)
    assert negative_score.upper == pytest.approx(-0.69)
    assert positive_score.prototype_digest == artifact.digest()
    assert positive_score.query_vector_digest == positive_query.digest()


def test_interval_distance_encloses_every_endpoint_possibility() -> None:
    space, _, _, _, artifact = support()
    query = vector("query-wide", (0.2, 0.8), (0.2, 0.8), space=space)
    score = contrastive_margin(query, artifact, space)
    assert score.lower == pytest.approx(-0.6)
    assert score.upper == pytest.approx(0.6)
    assert ContrastiveMargin(
        query.digest(), artifact.digest(), score.lower, score.upper
    ).digest() == score.digest()


@pytest.mark.parametrize(
    ("query_name", "angle", "balance", "expected"),
    [
        ("present", 0.84, 0.86, Disposition.PRESENT),
        ("absent", 0.14, 0.16, Disposition.CERTIFIED_ABSENT),
        ("middle", 0.5, 0.5, Disposition.INDETERMINATE),
        ("wide", (0.2, 0.8), (0.2, 0.8), Disposition.INDETERMINATE),
    ],
)
def test_query_evaluation_has_interval_safe_dispositions(
    query_name: str,
    angle: float | tuple[float, float],
    balance: float | tuple[float, float],
    expected: Disposition,
) -> None:
    space, _, _, _, artifact = support()
    result = evaluate_support_prototype(
        formula(space, artifact),
        artifact,
        space,
        vector("query-" + query_name, angle, balance, space=space),
    )
    assert result.disposition is expected
    assert result.uncertainty is not None
    if expected is Disposition.PRESENT:
        assert result.unwrap() is True
    elif expected is Disposition.CERTIFIED_ABSENT:
        assert result.certificate.startswith("operational-contrastive-nonmatch:")


def test_upstream_nonpresent_features_never_become_negative_by_default() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    absent = Evidence.certified_absent(origin("absent"), "no feature packet")
    indeterminate = Evidence.indeterminate(origin("unknown"), "blurred panel")
    error = Evidence.error(origin("error"), "ExtractorCrash", "GPU failed")
    assert evaluate_support_prototype(
        predicate, artifact, space, absent
    ).disposition is Disposition.INDETERMINATE
    assert evaluate_support_prototype(
        predicate, artifact, space, indeterminate
    ).disposition is Disposition.INDETERMINATE
    propagated = evaluate_support_prototype(predicate, artifact, space, error)
    assert propagated.disposition is Disposition.ERROR
    assert propagated.error_type == "ExtractorCrash"


def test_present_feature_evidence_preserves_upstream_provenance() -> None:
    space, _, _, _, artifact = support()
    packet = vector("query-provenance", 0.84, 0.86, space=space)
    upstream = Evidence.present(packet, origin("admitted"))
    result = evaluate_support_prototype(
        formula(space, artifact), artifact, space, upstream
    )
    assert result.disposition is Disposition.PRESENT
    assert origin("admitted").digest() in result.provenance.input_digests


def test_integrity_failures_are_errors_not_negative_predictions() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    query = vector("query-integrity", 0.84, 0.86, space=space)
    wrong_formula = replace(predicate, prototype_digest=digest("wrong artifact"))
    result = evaluate_support_prototype(wrong_formula, artifact, space, query)
    assert result.disposition is Disposition.ERROR
    assert result.disposition is not Disposition.CERTIFIED_ABSENT
    assert result.error_type == "SupportPrototypeIntegrityError"


def test_query_cannot_reuse_a_support_panel() -> None:
    space, _, positive, _, artifact = support()
    result = evaluate_support_prototype(
        formula(space, artifact), artifact, space, positive[0]
    )
    assert result.disposition is Disposition.ERROR
    assert "overlaps frozen support" in result.reason


def test_static_formula_validation_rejects_cross_artifact_splicing() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    altered_artifact = replace(
        artifact,
        support_assignment_digest=digest("different panel-side assignment"),
    )
    with pytest.raises(SupportPrototypeIntegrityError, match="prototype artifact"):
        validate_prototype_formula(predicate, altered_artifact, space)


def test_registered_support_prototype_executes_through_closed_ir() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    registry = LegRegistry()
    reference = register_support_prototype_leg(
        registry, predicate, artifact, space
    )
    registry.freeze()
    atom = Atom(
        StaticLegCall(reference, ("features",)),
        Relation.PRESENT,
        predicate.claim,
    )
    boundary = {"features": SUPPORT_PROTOTYPE_FEATURES}
    validate_formula(atom, registry, boundary)

    positive = evaluate_formula(
        atom,
        registry,
        {
            "features": TypedValue(
                SUPPORT_PROTOTYPE_FEATURES,
                vector("registered-positive", 0.84, 0.86, space=space),
            )
        },
    )
    negative = evaluate_formula(
        atom,
        registry,
        {
            "features": TypedValue(
                SUPPORT_PROTOTYPE_FEATURES,
                vector("registered-negative", 0.14, 0.16, space=space),
            )
        },
    )
    assert positive.disposition is Disposition.PRESENT
    assert negative.disposition is Disposition.CERTIFIED_ABSENT


def test_registered_leg_identity_binds_formula_and_prototypes() -> None:
    space, _, _, _, artifact = support()
    predicate = formula(space, artifact)
    first = LegRegistry()
    first_reference = register_support_prototype_leg(
        first, predicate, artifact, space
    )
    changed = replace(predicate, decision_margin=0.2)
    second = LegRegistry()
    second_reference = register_support_prototype_leg(
        second, changed, artifact, space
    )
    assert first_reference.contract_digest != second_reference.contract_digest
    assert (
        first.resolve(first_reference).operational_digest
        != second.resolve(second_reference).operational_digest
    )
