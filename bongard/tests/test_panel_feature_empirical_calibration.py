from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

import bongard.panel_feature_empirical_calibration as c
import bongard.panel_feature_observation as f
import bongard.panel_soft_ontology as o


def _d(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _spec(kind: o.GestaltKind) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.GESTALT_RESEMBLANCE,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.GestaltResemblanceParameters(kind),
    )


def _domain(*specs: o.PanelFeatureSpec) -> o.FeatureDomain:
    ordered = tuple(sorted(specs, key=lambda item: item.spec_digest))
    first = ordered[0]
    return o.FeatureDomain(
        first.family,
        first.subject_scope,
        first.reference_frame,
        ordered,
    )


def _observation(
    *,
    panel_digest: str,
    observed_spec: o.PanelFeatureSpec | None,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    resolution: f.BindingResolution = f.BindingResolution.COMPLETE,
    issue: f.ObservationIssue | None = None,
) -> f.PanelFeatureObservationSet:
    target_axis_spec = _spec(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(target_axis_spec)
    context = f.panel_only_observation_inventory(
        panel_digest=panel_digest,
        observer_contract_digest=observer_contract_digest,
        panel_context_receipt_digest=_d(f"context:{panel_digest}"),
    )
    binding = f.eligible_axis_bindings(axis, context)[0]
    observed = () if observed_spec is None else (observed_spec,)
    points = () if observed_spec is None else (o.QuantizedPoint(8, 8),)
    row = f.BindingFeatureObservation(
        axis.axis_digest,
        binding,
        resolution,
        observed,
        points,
        issue,
        _d(f"observation:{panel_digest}"),
    )
    axis_observation = f.PanelAxisObservation(
        context,
        axis,
        observer_contract_digest,
        measurement_protocol_digest,
        (row,),
    )
    return f.PanelFeatureObservationSet(
        panel_digest,
        observer_contract_digest,
        measurement_protocol_digest,
        (axis_observation,),
    )


def _case(
    *,
    case_id: str,
    panel_digest: str,
    spec: o.PanelFeatureSpec,
    truth: c.FeatureCalibrationTruth,
    nonce: str,
    annotation_protocol_digest: str,
) -> c.HeldOutFeatureCalibrationCase:
    return c.HeldOutFeatureCalibrationCase(
        case_id,
        panel_digest,
        spec.spec_digest,
        "val",
        f"cluster-{case_id}",
        c.feature_calibration_label_commitment(
            case_id=case_id,
            panel_digest=panel_digest,
            spec_digest=spec.spec_digest,
            annotation_protocol_digest=annotation_protocol_digest,
            truth=truth,
            label_nonce_digest=nonce,
        ),
    )


def _plan_and_measurements(
    *, accepted_error_ppm: int = 1_000_000
) -> tuple[
    c.FeatureObservationCalibrationPlan,
    tuple[c.HeldOutLabeledFeatureObservation, ...],
]:
    bird = _spec(o.GestaltKind.BIRD_LIKE)
    tool = _spec(o.GestaltKind.TOOL_LIKE)
    contract = _d("observer-contract")
    protocol = _d("measurement-protocol")
    annotation = _d("annotation-protocol")
    present_panel = _d("present-panel")
    absent_panel = _d("absent-panel")
    present_nonce = _d("present-label-secret")
    absent_nonce = _d("absent-label-secret")
    cases = (
        _case(
            case_id="case-0001",
            panel_digest=present_panel,
            spec=bird,
            truth=c.FeatureCalibrationTruth.PRESENT,
            nonce=present_nonce,
            annotation_protocol_digest=annotation,
        ),
        _case(
            case_id="case-0002",
            panel_digest=absent_panel,
            spec=bird,
            truth=c.FeatureCalibrationTruth.ABSENT,
            nonce=absent_nonce,
            annotation_protocol_digest=annotation,
        ),
    )
    plan = c.FeatureObservationCalibrationPlan(
        _domain(bird),
        contract,
        protocol,
        annotation,
        _d("corpus-manifest"),
        _d("split-manifest"),
        _d("exposure-ledger"),
        _d("holdout-selection-receipt"),
        cases,
        950_000,
        accepted_error_ppm,
        accepted_error_ppm,
        1,
        1,
        1_700_000_000,
        1_800_000_000,
    )
    measurements = (
        c.HeldOutLabeledFeatureObservation.create(
            plan,
            case_id="case-0001",
            observation_set=_observation(
                panel_digest=present_panel,
                observed_spec=bird,
                observer_contract_digest=contract,
                measurement_protocol_digest=protocol,
            ),
            truth=c.FeatureCalibrationTruth.PRESENT,
            label_nonce_digest=present_nonce,
            annotation_receipt_digest=_d("present-annotation-receipt"),
        ),
        c.HeldOutLabeledFeatureObservation.create(
            plan,
            case_id="case-0002",
            observation_set=_observation(
                panel_digest=absent_panel,
                observed_spec=tool,
                observer_contract_digest=contract,
                measurement_protocol_digest=protocol,
            ),
            truth=c.FeatureCalibrationTruth.ABSENT,
            label_nonce_digest=absent_nonce,
            annotation_receipt_digest=_d("absent-annotation-receipt"),
        ),
    )
    return plan, measurements


def test_exact_binomial_upper_bound_rounds_outward_on_ppm_grid() -> None:
    assert (
        c.one_sided_binomial_error_upper_ppm(
            errors=0, trials=1, confidence_ppm=950_000
        )
        == 950_000
    )
    assert (
        c.one_sided_binomial_error_upper_ppm(
            errors=0, trials=2, confidence_ppm=950_000
        )
        == 776_394
    )
    assert (
        c.one_sided_binomial_error_upper_ppm(
            errors=2, trials=2, confidence_ppm=950_000
        )
        == 1_000_000
    )
    with pytest.raises(c.PanelFeatureEmpiricalCalibrationError, match="errors/trials"):
        c.one_sided_binomial_error_upper_ppm(
            errors=2, trials=1, confidence_ppm=950_000
        )


def test_complete_held_out_manifest_yields_empirical_grants_but_no_authority() -> None:
    plan, measurements = _plan_and_measurements()
    assert c.FeatureObservationCalibrationPlan.from_data(plan.to_data()) == plan
    assert (
        c.HeldOutLabeledFeatureObservation.from_data(
            measurements[0].to_data(), plan=plan
        )
        == measurements[0]
    )

    outcome = c.score_feature_observation_calibration(plan, measurements)
    assert outcome.match_claim_count == 1
    assert outcome.false_positive_count == 0
    assert outcome.nonmatch_claim_count == 1
    assert outcome.false_negative_count == 0
    assert outcome.indeterminate_count == 0
    assert outcome.error_count == 0
    assert outcome.presence_assessment is not None
    assert outcome.absence_claim_assessment is not None
    assert outcome.presence_assessment.assessed_error_upper_ppm == 950_000
    assert outcome.gaps == ()
    assert (
        c.cold_replay_feature_observation_calibration(
            plan, measurements, outcome
        )
        == outcome
    )

    presence_only = c.derive_empirical_feature_calibration_grants(plan, outcome)
    assert type(presence_only.presence_grant) is o.PresenceCalibrationGrant
    assert presence_only.absence_grant is None
    assert {
        item.kind for item in presence_only.gaps
    } == {c.FeatureCalibrationGapKind.MISSING_INVENTORY_COMPLETENESS_CALIBRATION}
    assert presence_only.to_data()["feature_calibration_authority_issued"] is False
    assert presence_only.to_data()["scientific_projection_authorized"] is False

    inventory_assessment = o.CalibrationAssessment(
        o.CalibrationRisk.OWNER_INVENTORY_OMISSION,
        _d("independent-inventory-population"),
        _d("independent-inventory-annotation-protocol"),
        100,
        1_000_000,
        100_000,
        990_000,
        plan.valid_from_unix,
        plan.valid_through_unix,
        _d("inventory-assessment-receipt"),
    )
    prerequisites = c.AbsenceCalibrationPrerequisites(
        plan.domain.domain_digest,
        plan.observer_contract_digest,
        _d("owner-enumeration-protocol"),
        _d("search-protocol"),
        inventory_assessment,
        o.EnumerationResolution.GRID16_FULL_PANEL,
        tuple(sorted(o.RejectionKind, key=lambda item: item.value)),
        _d("inventory-calibration-receipt"),
    )
    grants = c.derive_empirical_feature_calibration_grants(
        plan, outcome, absence_prerequisites=prerequisites
    )
    assert type(grants.presence_grant) is o.PresenceCalibrationGrant
    assert type(grants.absence_grant) is o.AbsenceCalibrationGrant
    assert grants.gaps == ()

    # Compatibility is structural.  Only this explicit external caller supplies
    # an authority identity, trust root, and issuance receipt.  The calibration
    # module itself never does so.
    authority = o.FeatureCalibrationAuthority(
        "external.calibration.authority",
        plan.domain,
        plan.observer_contract_digest,
        _d("external-trust-root"),
        _d("external-authority-issuance-receipt"),
        grants.presence_grant,
        grants.absence_grant,
    )
    assert authority.presence_grant == grants.presence_grant
    assert authority.absence_grant == grants.absence_grant


def test_partial_manifest_and_low_claim_coverage_remain_typed_gaps() -> None:
    plan, measurements = _plan_and_measurements()
    partial = c.score_feature_observation_calibration(plan, measurements[:1])
    assert partial.presence_assessment is None
    assert partial.absence_claim_assessment is None
    assert partial.missing_case_ids == ("case-0002",)
    assert {
        item.kind for item in partial.gaps
    } == {c.FeatureCalibrationGapKind.INCOMPLETE_HELD_OUT_MANIFEST}

    contract = plan.observer_contract_digest
    protocol = plan.observer_measurement_protocol_digest
    unclear_measurements = tuple(
        c.HeldOutLabeledFeatureObservation.create(
            plan,
            case_id=item.case_id,
            observation_set=_observation(
                panel_digest=plan.case(item.case_id).panel_digest,
                observed_spec=None,
                observer_contract_digest=contract,
                measurement_protocol_digest=protocol,
                resolution=f.BindingResolution.UNCLEAR,
                issue=f.ObservationIssue.AMBIGUOUS_GEOMETRY,
            ),
            truth=item.truth,
            label_nonce_digest=item.label_nonce_digest,
            annotation_receipt_digest=item.annotation_receipt_digest,
        )
        for item in measurements
    )
    unclear = c.score_feature_observation_calibration(plan, unclear_measurements)
    assert unclear.indeterminate_count == 2
    assert unclear.match_claim_count == 0
    assert unclear.nonmatch_claim_count == 0
    assert {
        (item.kind, item.risk) for item in unclear.gaps
    } == {
        (
            c.FeatureCalibrationGapKind.INSUFFICIENT_DECISIVE_CLAIMS,
            o.CalibrationRisk.FALSE_POSITIVE_CLAIM,
        ),
        (
            c.FeatureCalibrationGapKind.INSUFFICIENT_DECISIVE_CLAIMS,
            o.CalibrationRisk.FALSE_NEGATIVE_CLAIM,
        ),
    }


def test_failed_error_bound_is_a_gap_not_a_favorable_assessment() -> None:
    plan, measurements = _plan_and_measurements(accepted_error_ppm=100_000)
    outcome = c.score_feature_observation_calibration(plan, measurements)
    assert outcome.presence_assessment is None
    assert outcome.absence_claim_assessment is None
    assert {
        item.kind for item in outcome.gaps
    } == {c.FeatureCalibrationGapKind.ERROR_BOUND_EXCEEDED}
    assert all(
        item.assessed_error_upper_ppm == 950_000 for item in outcome.gaps
    )
    grants = c.derive_empirical_feature_calibration_grants(plan, outcome)
    assert grants.presence_grant is None
    assert grants.absence_grant is None


def test_label_and_observation_custody_cannot_be_rewritten_after_plan() -> None:
    plan, measurements = _plan_and_measurements()
    original = measurements[0]
    with pytest.raises(c.PanelFeatureEmpiricalCalibrationError, match="commitment"):
        c.HeldOutLabeledFeatureObservation.create(
            plan,
            case_id=original.case_id,
            observation_set=original.observation_set,
            truth=c.FeatureCalibrationTruth.ABSENT,
            label_nonce_digest=original.label_nonce_digest,
            annotation_receipt_digest=original.annotation_receipt_digest,
        )
    wrong_panel = _observation(
        panel_digest=_d("wrong-panel"),
        observed_spec=_spec(o.GestaltKind.BIRD_LIKE),
        observer_contract_digest=plan.observer_contract_digest,
        measurement_protocol_digest=plan.observer_measurement_protocol_digest,
    )
    with pytest.raises(c.PanelFeatureEmpiricalCalibrationError, match="custody"):
        c.HeldOutLabeledFeatureObservation.create(
            plan,
            case_id=original.case_id,
            observation_set=wrong_panel,
            truth=original.truth,
            label_nonce_digest=original.label_nonce_digest,
            annotation_receipt_digest=original.annotation_receipt_digest,
        )

    raw = deepcopy(original.to_data())
    raw["truth"] = "absent"
    with pytest.raises(c.PanelFeatureEmpiricalCalibrationError, match="commitment"):
        c.HeldOutLabeledFeatureObservation.from_data(raw, plan=plan)


def test_plan_rejects_pseudoreplicated_panels_or_dependence_clusters() -> None:
    plan, _ = _plan_and_measurements()
    first, second = plan.cases
    duplicate_panel = c.HeldOutFeatureCalibrationCase(
        second.case_id,
        first.panel_digest,
        second.spec_digest,
        second.split,
        second.dependence_cluster_id,
        second.label_commitment_digest,
    )
    with pytest.raises(
        c.PanelFeatureEmpiricalCalibrationError, match="panel sampling unit"
    ):
        c.FeatureObservationCalibrationPlan(
            plan.domain,
            plan.observer_contract_digest,
            plan.observer_measurement_protocol_digest,
            plan.annotation_protocol_digest,
            plan.corpus_manifest_digest,
            plan.split_manifest_digest,
            plan.exposure_ledger_digest,
            plan.holdout_selection_receipt_digest,
            (first, duplicate_panel),
            plan.confidence_ppm,
            plan.accepted_false_positive_upper_ppm,
            plan.accepted_false_negative_upper_ppm,
            plan.minimum_presence_claim_count,
            plan.minimum_absence_claim_count,
            plan.valid_from_unix,
            plan.valid_through_unix,
        )

