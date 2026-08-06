from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import (
    Atom,
    IRValidationError,
    Quantity,
    Relation,
    StaticLegCall,
    evaluate_formula,
    validate_formula,
)
from bongard.legs import (
    FROZEN_VISUAL_SCORE,
    PANEL,
    SOFT_SEMANTIC,
    AffirmativeRelation,
    LegSemantics,
    LegRegistry,
    TypedValue,
    Unit,
)
from bongard.soft_predicates import (
    CalibrationDesign,
    CalibrationError,
    CalibrationObservation,
    CalibratedPredictiveSupport,
    DevelopmentUnit,
    FrozenVisualScore,
    ObservationRole,
    PreregisteredCalibrationPlan,
    SoftPredicateClaim,
    SoftPredicateIntegrityError,
    fit_monotone_calibration,
    register_soft_predicate,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def claim() -> SoftPredicateClaim:
    return SoftPredicateClaim(
        phrase="bird-like object",
        affirmative_cues=("beak-like protrusion", "paired wing-like regions"),
        model_id="gpt-vision-exact-2026-08-05",
        prompt_id="prompt:bird-cues:sha256:1111",
        decoder_id="strict-cue-score-v1",
    )


def design() -> CalibrationDesign:
    return CalibrationDesign(
        annotation_protocol_id="two-independent-raters-plus-adjudication-v1",
        annotation_protocol_digest=digest("annotation-protocol-bytes"),
        annotation_ontology_id="affirmative-bird-cues-v1",
        annotation_ontology_digest=digest("annotation-ontology-bytes"),
        target_population_id="shape-bongard-v2-held-out-tasks",
        population_manifest_digest=digest("target-population-manifest"),
        sampling_design_id="task-cluster-stratified-v1",
        sampling_design_digest=digest("sampling-design-bytes"),
        scorer_artifact_id="frozen-vision-scorer-v1",
        scorer_artifact_digest=digest("scorer-code-weights-config"),
        score_admission_protocol_id="external-score-admission-v1",
        score_admission_protocol_digest=digest("score-admission-protocol"),
    )


def development_units(
    *, clusters_per_bin: int = 100, panels_per_cluster: int = 2
) -> tuple[DevelopmentUnit, ...]:
    units = []
    for bin_index in range(2):
        for cluster_index in range(clusters_per_bin):
            cluster_id = f"cluster-b{bin_index}-c{cluster_index:03d}"
            for panel_index in range(panels_per_cluster):
                observation_id = (
                    f"b{bin_index}-c{cluster_index:03d}-p{panel_index:03d}"
                )
                units.append(
                    DevelopmentUnit(
                        observation_id=observation_id,
                        task_id=f"task-b{bin_index}-c{cluster_index:03d}",
                        group_id=f"group-b{bin_index}-c{cluster_index:03d}",
                        model_call_id=f"call-b{bin_index}-c{cluster_index:03d}",
                        cluster_id=cluster_id,
                        panel_digest=digest("development-panel-" + observation_id),
                    )
                )
    return tuple(sorted(units, key=lambda unit: unit.observation_id))


def plan(
    soft_claim: SoftPredicateClaim,
    *,
    minimum_clusters_per_bin: int = 20,
    threshold: float = 0.5,
    clusters_per_bin: int = 100,
    panels_per_cluster: int = 2,
) -> PreregisteredCalibrationPlan:
    return PreregisteredCalibrationPlan(
        verifier_id="bongard-verifier-v1",
        registration_id="bird-plan-published-before-query-v1",
        claim_digest=soft_claim.digest(),
        design=design(),
        development_units=development_units(
            clusters_per_bin=clusters_per_bin,
            panels_per_cluster=panels_per_cluster,
        ),
        bin_edges=(0.0, 0.5, 1.0),
        confidence_level=0.9,
        minimum_clusters_per_bin=minimum_clusters_per_bin,
        affirmative_threshold=threshold,
    )


def observations(
    soft_claim: SoftPredicateClaim,
    preregistration: PreregisteredCalibrationPlan,
    *,
    rates: tuple[float, float] = (0.1, 0.9),
) -> tuple[CalibrationObservation, ...]:
    result = []
    clusters_by_bin = [
        sorted(
            {
                unit.cluster_id
                for unit in preregistration.development_units
                if unit.observation_id.startswith(f"b{bin_index}-")
            }
        )
        for bin_index in range(2)
    ]
    affirmative_clusters = [
        set(cluster_ids[: round(rate * len(cluster_ids))])
        for cluster_ids, rate in zip(clusters_by_bin, rates, strict=True)
    ]
    for unit in preregistration.development_units:
        bin_index = int(unit.observation_id[1])
        result.append(
            CalibrationObservation(
                observation_id=unit.observation_id,
                task_id=unit.task_id,
                group_id=unit.group_id,
                model_call_id=unit.model_call_id,
                cluster_id=unit.cluster_id,
                panel_digest=unit.panel_digest,
                claim_digest=soft_claim.digest(),
                model_id=soft_claim.model_id,
                prompt_id=soft_claim.prompt_id,
                decoder_id=soft_claim.decoder_id,
                scorer_artifact_digest=preregistration.design.scorer_artifact_digest,
                admitting_verifier_id=preregistration.verifier_id,
                score_admission_protocol_digest=(
                    preregistration.design.score_admission_protocol_digest
                ),
                score_admission_receipt_digest=digest(
                    "development-score-receipt-" + unit.observation_id
                ),
                annotation_protocol_digest=(
                    preregistration.design.annotation_protocol_digest
                ),
                annotation_ontology_digest=(
                    preregistration.design.annotation_ontology_digest
                ),
                annotation_receipt_digest=digest(
                    "development-annotation-receipt-" + unit.observation_id
                ),
                role=ObservationRole.DEVELOPMENT,
                score=(0.25, 0.75)[bin_index],
                affirmative_label=unit.cluster_id
                in affirmative_clusters[bin_index],
            )
        )
    return tuple(result)


def calibrated(
    *,
    rates: tuple[float, float] = (0.1, 0.9),
    clusters_per_bin: int = 100,
    panels_per_cluster: int = 2,
):
    soft_claim = claim()
    preregistration = plan(
        soft_claim,
        clusters_per_bin=clusters_per_bin,
        panels_per_cluster=panels_per_cluster,
    )
    artifact = fit_monotone_calibration(
        preregistration,
        soft_claim,
        observations(soft_claim, preregistration, rates=rates),
        expected_plan_digest=preregistration.digest(),
    )
    return soft_claim, preregistration, artifact


def registered(*, rates: tuple[float, float] = (0.1, 0.9)):
    soft_claim, preregistration, artifact = calibrated(rates=rates)
    registry = LegRegistry()
    handle = register_soft_predicate(
        registry,
        name="bird_like_calibrated",
        version="1",
        claim=soft_claim,
        calibration=artifact,
        expected_claim_digest=soft_claim.digest(),
        expected_calibration_digest=artifact.digest(),
    )
    registry.freeze()
    return registry, handle, soft_claim, preregistration, artifact


def packet(
    soft_claim: SoftPredicateClaim,
    preregistration: PreregisteredCalibrationPlan,
    *,
    score: float | None,
    panel: str = "fresh-query-panel",
    missing_reason: str | None = None,
    **changes: object,
) -> FrozenVisualScore:
    fields: dict[str, object] = {
        "task_id": "held-out-query-task",
        "group_id": "held-out-query-group",
        "model_call_id": "held-out-query-call",
        "cluster_id": "held-out-query-cluster",
        "panel_digest": digest(panel),
        "claim_digest": soft_claim.digest(),
        "model_id": soft_claim.model_id,
        "prompt_id": soft_claim.prompt_id,
        "decoder_id": soft_claim.decoder_id,
        "scorer_artifact_digest": preregistration.design.scorer_artifact_digest,
        "admitting_verifier_id": preregistration.verifier_id,
        "score_admission_protocol_digest": (
            preregistration.design.score_admission_protocol_digest
        ),
        "score_admission_receipt_digest": digest(panel + "-receipt"),
        "score": score,
        "description": "a compact body with a beak-like protrusion",
        "observed_cue_ids": () if score is None else ("beak-like protrusion",),
        "missing_reason": missing_reason,
    }
    fields.update(changes)
    return FrozenVisualScore(**fields)  # type: ignore[arg-type]


def evaluate(registry: LegRegistry, handle: object, value: object):
    return evaluate_formula(
        handle.atom(),  # type: ignore[attr-defined]
        registry,
        {"frozen_score": TypedValue(FROZEN_VISUAL_SCORE, value)},
    )


def test_claim_and_calibration_are_content_addressed_and_monotone() -> None:
    soft_claim, preregistration, artifact = calibrated()
    assert len(soft_claim.digest()) == 64
    assert artifact.plan_digest == preregistration.digest()
    assert artifact.claim_digest == soft_claim.digest()
    assert [band.support_lower for band in artifact.bands] == sorted(
        band.support_lower for band in artifact.bands
    )
    assert [band.support_upper for band in artifact.bands] == sorted(
        band.support_upper for band in artifact.bands
    )
    assert artifact.bands[0].support_upper < 0.5
    assert artifact.bands[1].support_lower > 0.5


def test_plan_preregisters_exact_design_and_development_membership() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim, clusters_per_bin=20)
    data = preregistration.to_data()
    assert data["design"] == preregistration.design.to_data()
    assert data["development_units"] == [
        unit.to_data() for unit in preregistration.development_units
    ]
    assert data["development_manifest_digest"] == (
        preregistration.development_manifest_digest
    )

    changed_sampling = replace(
        preregistration.design,
        sampling_design_digest=digest("different-sampling-design"),
    )
    assert replace(preregistration, design=changed_sampling).digest() != (
        preregistration.digest()
    )
    changed_ontology = replace(
        preregistration.design,
        annotation_ontology_digest=digest("different-ontology"),
    )
    assert replace(preregistration, design=changed_ontology).digest() != (
        preregistration.digest()
    )
    changed_scorer = replace(
        preregistration.design,
        scorer_artifact_digest=digest("different-scorer-artifact"),
    )
    assert replace(preregistration, design=changed_scorer).digest() != (
        preregistration.digest()
    )


def test_fit_requires_exact_preregistered_units_and_design_receipts() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim, clusters_per_bin=20)
    development = observations(soft_claim, preregistration)

    with pytest.raises(CalibrationError, match="exact preregistered"):
        fit_monotone_calibration(
            preregistration,
            soft_claim,
            development[:-1],
            expected_plan_digest=preregistration.digest(),
        )

    wrong_unit = (
        replace(development[0], task_id="substituted-task"),
    ) + development[1:]
    with pytest.raises(SoftPredicateIntegrityError, match="task/group/call"):
        fit_monotone_calibration(
            preregistration,
            soft_claim,
            wrong_unit,
            expected_plan_digest=preregistration.digest(),
        )

    for field, value in (
        ("scorer_artifact_digest", digest("substituted-scorer")),
        ("admitting_verifier_id", "substituted-verifier"),
        (
            "score_admission_protocol_digest",
            digest("substituted-score-admission-protocol"),
        ),
        ("annotation_protocol_digest", digest("substituted-protocol")),
        ("annotation_ontology_digest", digest("substituted-ontology")),
    ):
        wrong_design = (replace(development[0], **{field: value}),) + development[1:]
        with pytest.raises(SoftPredicateIntegrityError, match="operational identity"):
            fit_monotone_calibration(
                preregistration,
                soft_claim,
                wrong_design,
                expected_plan_digest=preregistration.digest(),
            )


def test_task_group_or_model_call_cannot_be_split_into_fake_clusters() -> None:
    soft_claim = claim()
    preregistration = plan(
        soft_claim, clusters_per_bin=20, panels_per_cluster=2
    )
    units = list(preregistration.development_units)
    assert units[0].task_id == units[1].task_id
    units[1] = replace(units[1], cluster_id="fake-independent-cluster")
    with pytest.raises(ValueError, match="split across dependence clusters"):
        replace(preregistration, development_units=tuple(units))


def test_repeated_panels_do_not_shrink_cluster_level_bounds() -> None:
    _, _, one_panel = calibrated(
        clusters_per_bin=40, panels_per_cluster=1
    )
    _, _, twenty_panels = calibrated(
        clusters_per_bin=40, panels_per_cluster=20
    )
    for one, twenty in zip(one_panel.bands, twenty_panels.bands, strict=True):
        assert one.cluster_count == twenty.cluster_count == 40
        assert twenty.panel_count == 20 * one.panel_count
        assert twenty.cluster_support_mean == one.cluster_support_mean
        assert (twenty.support_lower, twenty.support_upper) == (
            one.support_lower,
            one.support_upper,
        )
    radius = math.sqrt(math.log(40.0) / (2.0 * 40.0))
    assert one_panel.bands[0].support_upper == pytest.approx(0.1 + radius)
    assert one_panel.bands[1].support_lower == pytest.approx(0.9 - radius)


def test_registered_leg_binds_claim_calibration_and_fixed_positive_atom() -> None:
    registry, handle, soft_claim, _, artifact = registered()
    formula = handle.atom("image")
    assert handle.contract.domain == (FROZEN_VISUAL_SCORE,)
    assert handle.contract.codomain == SOFT_SEMANTIC
    assert handle.contract.semantics is LegSemantics.DERIVED
    assert handle.contract.affirmative_relations == frozenset(
        {AffirmativeRelation.AT_LEAST}
    )
    assert formula.relation is Relation.AT_LEAST
    assert formula.lower == Quantity(artifact.affirmative_threshold, Unit.PROBABILITY)
    assert handle.contract.operational_digest == handle.operational_digest
    assert soft_claim.digest() in {
        handle._claim_digest,  # exact verifier commitment retained by handle
        artifact.claim_digest,
    }
    validate_formula(formula, registry, {"image": FROZEN_VISUAL_SCORE})
    with pytest.raises(IRValidationError, match="expected ValueType.*frozen_visual_score"):
        validate_formula(formula, registry, {"image": PANEL})


def test_result_is_named_calibrated_predictive_support_not_semantic_truth() -> None:
    registry, handle, soft_claim, preregistration, artifact = registered()
    raw = registry.invoke(
        handle.reference,
        (
            TypedValue(
                FROZEN_VISUAL_SCORE,
                packet(soft_claim, preregistration, score=0.75),
            ),
        ),
    )
    result = raw.unwrap().value
    assert isinstance(result, CalibratedPredictiveSupport)
    assert result.target_population_id == preregistration.design.target_population_id
    assert result.sampling_design_id == preregistration.design.sampling_design_id
    assert result.calibration_digest == artifact.digest()
    assert result.effective_cluster_count == artifact.bands[1].cluster_count
    with pytest.raises(TypeError, match="empirical evidence, not truth"):
        bool(result)


def test_raw_caller_score_is_not_treated_as_a_vision_observation() -> None:
    registry, handle, _, _, _ = registered()
    result = evaluate(registry, handle, 0.75)
    assert result.disposition is Disposition.ERROR
    assert result.error_type == "MalformedVisualScore"


def test_four_dispositions_do_not_collapse_failure_into_negative() -> None:
    registry, handle, soft_claim, preregistration, _ = registered()
    positive = evaluate(
        registry, handle, packet(soft_claim, preregistration, score=0.75)
    )
    negative = evaluate(
        registry,
        handle,
        packet(
            soft_claim,
            preregistration,
            score=0.25,
            panel="fresh-low-query",
        ),
    )
    missing = evaluate(
        registry,
        handle,
        packet(
            soft_claim,
            preregistration,
            score=None,
            panel="fresh-missing-query",
            missing_reason="vision response omitted its score",
        ),
    )
    malformed = evaluate(registry, handle, {"score": 0.75, "label": True})
    assert positive.disposition is Disposition.PRESENT
    assert negative.disposition is Disposition.CERTIFIED_ABSENT
    assert missing.disposition is Disposition.INDETERMINATE
    assert malformed.disposition is Disposition.ERROR


def test_upstream_prose_absence_cannot_bypass_soft_calibration() -> None:
    registry, handle, _, _, _ = registered()
    upstream = Evidence.certified_absent(
        Provenance("generic-vlm", "1", "free-text-absence"),
        "model says a cue is absent",
    )
    result = evaluate(registry, handle, upstream)
    assert result.disposition is Disposition.INDETERMINATE
    assert "calibrated score" in (result.reason or "")


def test_polarity_flip_and_query_label_rescue_are_rejected() -> None:
    registry, handle, soft_claim, _, _ = registered()
    flipped = Atom(
        call=StaticLegCall(handle.reference, ("panel",)),
        relation=Relation.AT_MOST,
        claim="negated bird-like support",
        lower=Quantity(handle.affirmative_threshold, Unit.PROBABILITY),
    )
    with pytest.raises(IRValidationError, match="not an affirmative orientation"):
        validate_formula(flipped, registry, {"panel": FROZEN_VISUAL_SCORE})

    # A query-time label is outside the frozen score DTO, not an invitation to
    # infer or reverse a class decision.
    result = evaluate(
        registry,
        handle,
        {"score": 0.1, "label": False, "claim_digest": soft_claim.digest()},
    )
    assert result.disposition is Disposition.ERROR


def test_calibration_rejects_query_or_support_leakage() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim)
    development = observations(soft_claim, preregistration)
    for leaked_role in (ObservationRole.QUERY, ObservationRole.SUPPORT):
        leaked = (replace(development[0], role=leaked_role),) + development[1:]
        with pytest.raises(CalibrationError, match="cannot fit calibration"):
            fit_monotone_calibration(
                preregistration,
                soft_claim,
                leaked,
                expected_plan_digest=preregistration.digest(),
            )


def test_query_cannot_reuse_any_development_dependence_identity() -> None:
    registry, handle, soft_claim, preregistration, artifact = registered()
    leaked_panel_digest = artifact.development_panel_digests[0]
    leaked = packet(
        soft_claim,
        preregistration,
        score=0.75,
        panel="placeholder",
        panel_digest=leaked_panel_digest,
    )
    result = evaluate(registry, handle, leaked)
    assert result.disposition is Disposition.ERROR
    assert "overlaps" in (result.reason or "")

    development_unit = preregistration.development_units[0]
    for field in ("task_id", "group_id", "model_call_id", "cluster_id"):
        identity_leak = packet(
            soft_claim,
            preregistration,
            score=0.75,
            panel="fresh-for-" + field,
            **{field: getattr(development_unit, field)},
        )
        assert evaluate(registry, handle, identity_leak).disposition is Disposition.ERROR


def test_calibration_and_query_operational_identity_must_match() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim)
    development = observations(soft_claim, preregistration)
    mismatched = (
        replace(development[0], decoder_id="opposite-polarity-decoder"),
    ) + development[1:]
    with pytest.raises(SoftPredicateIntegrityError, match="wrong operational identity"):
        fit_monotone_calibration(
            preregistration,
            soft_claim,
            mismatched,
            expected_plan_digest=preregistration.digest(),
        )

    registry, handle, _, query_plan, _ = registered()
    wrong_query = packet(
        soft_claim,
        query_plan,
        score=0.75,
        panel="wrong-model-query",
        model_id="different-model-version",
    )
    result = evaluate(registry, handle, wrong_query)
    assert result.disposition is Disposition.ERROR
    assert "wrong claim/scorer/admission" in (result.reason or "")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scorer_artifact_digest", digest("unregistered-scorer")),
        ("admitting_verifier_id", "caller-self-assertion"),
        ("score_admission_protocol_digest", digest("unregistered-admission")),
    ],
)
def test_query_score_requires_preregistered_scorer_and_admission_identity(
    field: str, value: str
) -> None:
    registry, handle, soft_claim, preregistration, _ = registered()
    unadmitted = packet(
        soft_claim,
        preregistration,
        score=0.75,
        panel="unadmitted-" + field,
        **{field: value},
    )
    result = evaluate(registry, handle, unadmitted)
    assert result.disposition is Disposition.ERROR
    assert "scorer/admission identity" in (result.reason or "")


def test_preregistration_digest_and_minimum_sample_are_enforced() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim, minimum_clusters_per_bin=20)
    with pytest.raises(SoftPredicateIntegrityError, match="preregistered digest"):
        fit_monotone_calibration(
            preregistration,
            soft_claim,
            observations(soft_claim, preregistration),
            expected_plan_digest=digest("not-the-plan"),
        )
    sparse_plan = plan(
        soft_claim,
        minimum_clusters_per_bin=20,
        clusters_per_bin=19,
    )
    with pytest.raises(CalibrationError, match="insufficient independent"):
        fit_monotone_calibration(
            sparse_plan,
            soft_claim,
            observations(soft_claim, sparse_plan),
            expected_plan_digest=sparse_plan.digest(),
        )


def test_nonmonotone_calibration_fails_instead_of_flipping_labels() -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim)
    with pytest.raises(CalibrationError, match="monotone direction"):
        fit_monotone_calibration(
            preregistration,
            soft_claim,
            observations(soft_claim, preregistration, rates=(1.0, 0.0)),
            expected_plan_digest=preregistration.digest(),
        )


@pytest.mark.parametrize("bad_score", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_malformed_scores_are_rejected_not_read_as_negative(bad_score: object) -> None:
    soft_claim = claim()
    preregistration = plan(soft_claim)
    with pytest.raises((TypeError, ValueError), match="query score"):
        packet(soft_claim, preregistration, score=bad_score)  # type: ignore[arg-type]

    registry, handle, _, query_plan, _ = registered()
    valid = packet(soft_claim, query_plan, score=0.75)
    object.__setattr__(valid, "score", bad_score)
    result = evaluate(registry, handle, valid)
    assert result.disposition is Disposition.ERROR


def test_finite_sample_interval_straddling_threshold_is_indeterminate() -> None:
    registry, handle, soft_claim, preregistration, _ = registered(
        rates=(0.5, 0.5)
    )
    result = evaluate(
        registry,
        handle,
        packet(soft_claim, preregistration, score=0.75),
    )
    assert result.disposition is Disposition.INDETERMINATE
    assert result.uncertainty is not None
    assert result.uncertainty.lower < handle.affirmative_threshold
    assert result.uncertainty.upper > handle.affirmative_threshold


def test_expected_claim_and_calibration_digests_cannot_be_substituted() -> None:
    soft_claim, _, artifact = calibrated()
    with pytest.raises(SoftPredicateIntegrityError, match="claim differs"):
        register_soft_predicate(
            LegRegistry(),
            name="bird_like_calibrated",
            version="1",
            claim=soft_claim,
            calibration=artifact,
            expected_claim_digest=digest("other-claim"),
            expected_calibration_digest=artifact.digest(),
        )
    with pytest.raises(SoftPredicateIntegrityError, match="calibration differs"):
        register_soft_predicate(
            LegRegistry(),
            name="bird_like_calibrated",
            version="1",
            claim=soft_claim,
            calibration=artifact,
            expected_claim_digest=soft_claim.digest(),
            expected_calibration_digest=digest("other-calibration"),
        )


def test_post_registration_configuration_tampering_becomes_error() -> None:
    registry, handle, soft_claim, preregistration, artifact = registered()
    frozen_formula = handle.atom()
    object.__setattr__(artifact.bands[1], "support_lower", 0.0)
    result = evaluate_formula(
        frozen_formula,
        registry,
        {
            "frozen_score": TypedValue(
                FROZEN_VISUAL_SCORE,
                packet(soft_claim, preregistration, score=0.75),
            )
        },
    )
    assert result.disposition is Disposition.ERROR
    assert result.error_type == "SoftPredicateIntegrityError"
    assert "calibration changed" in (result.reason or "")


def test_post_registration_source_tampering_invalidates_reference() -> None:
    registry, handle, soft_claim, preregistration, _ = registered()

    def tampered(_: object) -> Evidence[bool]:
        provenance = Provenance("attacker", "1", "polarity-flip")
        return Evidence.certified_absent(provenance, "fabricated negative")

    object.__setattr__(handle.contract, "implementation", tampered)
    with pytest.raises(ValueError, match="implementation changed"):
        evaluate(
            registry,
            handle,
            packet(soft_claim, preregistration, score=0.75),
        )
