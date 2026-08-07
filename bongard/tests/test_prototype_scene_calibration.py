from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json

import pytest

from bongard.canonical import canonical_digest
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import load_historical_exposure
from bongard.prototype_pair_cohort import (
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OPAQUE_TAG_IDS,
    plan_prototype_pair_cohort,
    prototype_pair_seed_commitment,
    task_id_inventory_digest,
)
from bongard.prototype_scene_calibration import (
    CalibrationDirection,
    PrototypeSceneCalibrationError,
    PrototypeSceneCalibrationFamily,
    PrototypeSceneCalibrationObservation,
    PrototypeSceneCalibrationPlan,
    PrototypeSceneDisposition,
    PrototypeSceneEvaluationContext,
    PrototypeSceneScoreStatus,
    PrototypeSceneTagScore,
    PrototypeSceneTagThreshold,
    assess_prototype_scene_calibration,
    create_prototype_scene_calibration_plan,
    evaluate_prototype_scene_score,
    fit_prototype_scene_calibration_family,
    threshold_commitment,
    verify_prototype_scene_calibration_family,
)
from bongard.release import OfficialReleaseDescriptor


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _cohort():
    historical = load_historical_exposure()
    shapes = list(historical.unused_basic_shape_families)
    target_a, target_b = shapes[:2]
    partners = shapes[2:30]
    tasks = {
        f"bd_{target_a}_0000",
        f"bd_{target_b}_0000",
        f"bd_{target_a}-{target_b}_0000",
    }
    tasks.update(f"bd_{target_a}-{shape}_0000" for shape in partners[:14])
    tasks.update(f"bd_{target_b}-{shape}_0000" for shape in partners[14:28])
    inventory = tuple(sorted(tasks))
    split_bytes = json.dumps(
        {"test": [], "train": list(inventory), "val": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    split_digest = "sha256:" + hashlib.sha256(split_bytes).hexdigest()
    release = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-calibration-test",
        archive_filename="ShapeBongard_V2.zip",
        archive_sha256="sha256:" + "a" * 64,
        archive_size_bytes=1,
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=split_digest,
        split_size_bytes=len(split_bytes),
        upstream_repository=OFFICIAL_UPSTREAM_REPOSITORY,
        upstream_commit=OFFICIAL_UPSTREAM_COMMIT,
        family_counts=(("bd", len(inventory)),),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=task_id_inventory_digest(inventory),
        corpus_manifest_sha256="sha256:" + "b" * 64,
    )
    exposure = ExposureLedger.create(release.corpus_manifest_sha256)
    seed = "prototype-scene-calibration-cohort-test"
    return plan_prototype_pair_cohort(
        release_descriptor=release,
        split_bytes=split_bytes,
        task_ids=inventory,
        exposure_predecessor=exposure,
        historical_seed=historical,
        selection_seed=seed,
        expected_seed_commitment=prototype_pair_seed_commitment(seed),
        expected_release_descriptor_digest=release.digest,
        expected_corpus_manifest_digest=release.corpus_manifest_sha256,
        expected_split_source_digest=release.split_sha256,
        expected_task_inventory_digest=release.task_ids_sha256,
        expected_exposure_predecessor_digest=exposure.digest,
        expected_historical_seed_digest=historical.seed_digest,
        expected_resolver_policy_digest=semantic_resolver_policy_digest(historical),
        expected_basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        expected_basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
    )


def _calibration_plan():
    cohort = _cohort()
    thresholds = (
        PrototypeSceneTagThreshold(OPAQUE_TAG_IDS[0], 300_000, 700_000),
        PrototypeSceneTagThreshold(OPAQUE_TAG_IDS[1], 300_000, 700_000),
    )
    catalog = _address("rubric-description-catalog")
    reference = _address("prototype-reference-catalog")
    protocol = _address("prototype-scene-protocol")
    model_identity = _address("reported-model-identity")
    environment = _address("observer-environment")
    plan = create_prototype_scene_calibration_plan(
        cohort_plan=cohort,
        thresholds=thresholds,
        description_catalog_digest=catalog,
        prototype_reference_digest=reference,
        observer_protocol_id="prototype.scene.observer.v1",
        observer_protocol_digest=protocol,
        model_id="gpt-test-fixed",
        model_identity_digest=model_identity,
        environment_digest=environment,
        expected_cohort_plan_digest=cohort.record_digest,
        expected_threshold_commitment=threshold_commitment(thresholds),
        expected_description_catalog_digest=catalog,
        expected_prototype_reference_digest=reference,
        expected_observer_protocol_digest=protocol,
        expected_model_identity_digest=model_identity,
        expected_environment_digest=environment,
    )
    return cohort, plan


def _score(tag_id: str, expected: str) -> PrototypeSceneTagScore:
    if expected == "present":
        return PrototypeSceneTagScore(
            tag_id,
            PrototypeSceneScoreStatus.SCORE,
            800_000,
            900_000,
            "scored",
            None,
        )
    return PrototypeSceneTagScore(
        tag_id,
        PrototypeSceneScoreStatus.SCORE,
        100_000,
        200_000,
        "scored",
        None,
    )


def _observations(plan: PrototypeSceneCalibrationPlan):
    result = []
    for scene in plan.scenes:
        expected = dict(scene.expected_tag_states)
        result.append(
            PrototypeSceneCalibrationObservation(
                calibration_plan_digest=plan.record_digest,
                cohort_plan_digest=plan.cohort_plan_digest,
                task_id=scene.task_id,
                panel_id=scene.panel_id,
                observer_artifact_digest=_address(
                    {"task": scene.task_id, "panel": scene.panel_id}
                ),
                observer_artifact_schema="gkm.prototype-scene-observer.v1",
                description_catalog_digest=plan.description_catalog_digest,
                prototype_reference_digest=plan.prototype_reference_digest,
                observer_protocol_id=plan.observer_protocol_id,
                observer_protocol_digest=plan.observer_protocol_digest,
                model_id=plan.model_id,
                model_identity_digest=plan.model_identity_digest,
                environment_digest=plan.environment_digest,
                observer_call_count=1,
                scores=tuple(
                    _score(tag_id, expected[tag_id]) for tag_id in OPAQUE_TAG_IDS
                ),
                adapter_protocol_id=(
                    "bongard.prototype-scene-observer/calibration-adapter-v1"
                ),
            )
        )
    return tuple(result)


def test_zero_error_accepts_exact_268752_and_cold_replays() -> None:
    cohort, plan = _calibration_plan()
    observations = _observations(plan)
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    assert assessment.all_four_bounds_accepted is True
    assert assessment.all_four_coverage_gates_accepted is True
    assert len(assessment.bounds) == 4
    assert all(item.cluster_count == 14 for item in assessment.bounds)
    assert all(item.error_cluster_count == 0 for item in assessment.bounds)
    assert all(item.abstention_cluster_count == 0 for item in assessment.bounds)
    assert all(item.coverage_gate_accepted for item in assessment.bounds)
    assert all(
        item.conditional_error_upper_ppm == 268_752
        for item in assessment.bounds
    )
    family = fit_prototype_scene_calibration_family(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    assert family.coverage_gate_accepted is True
    assert PrototypeSceneCalibrationPlan.from_data(plan.to_data()) == plan
    assert PrototypeSceneCalibrationFamily.from_data(family.to_data()) == family
    assert verify_prototype_scene_calibration_family(
        family.to_data(),
        calibration_plan=plan.to_data(),
        cohort_plan=cohort.to_data(),
        observations=[item.to_data() for item in observations],
        expected_family_digest=family.record_digest,
        expected_calibration_plan_digest=plan.record_digest,
        expected_cohort_plan_digest=cohort.record_digest,
    ) == family


def test_one_false_decision_fails_300k_tolerance() -> None:
    _cohort_plan, plan = _calibration_plan()
    observations = list(_observations(plan))
    scene_index = next(
        index
        for index, scene in enumerate(plan.scenes)
        if dict(scene.expected_tag_states)[OPAQUE_TAG_IDS[0]] == "present"
    )
    original = observations[scene_index]
    wrong = PrototypeSceneTagScore(
        OPAQUE_TAG_IDS[0],
        PrototypeSceneScoreStatus.SCORE,
        100_000,
        200_000,
        "scored",
        None,
    )
    observations[scene_index] = replace(
        original, scores=(wrong, original.scores[1])
    )
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    failed = next(
        item
        for item in assessment.bounds
        if item.tag_id == OPAQUE_TAG_IDS[0]
        and item.direction is CalibrationDirection.FALSE_ABSENT
    )
    assert failed.cluster_count == 14
    assert failed.error_cluster_count == 1
    assert failed.conditional_error_upper_ppm == 377_292
    assert failed.accepted is False
    with pytest.raises(PrototypeSceneCalibrationError, match="300000"):
        fit_prototype_scene_calibration_family(
            plan,
            observations,
            expected_calibration_plan_digest=plan.record_digest,
        )


@pytest.mark.parametrize(
    "status",
    (
        PrototypeSceneScoreStatus.MISSING,
        PrototypeSceneScoreStatus.ERROR,
        PrototypeSceneScoreStatus.PARSER_ERROR,
        PrototypeSceneScoreStatus.TRANSPORT_ERROR,
    ),
)
def test_technical_score_states_stay_in_denominator_and_count_as_errors(
    status: PrototypeSceneScoreStatus,
) -> None:
    _cohort_plan, plan = _calibration_plan()
    observations = list(_observations(plan))
    original = observations[0]
    failed_score = PrototypeSceneTagScore(
        OPAQUE_TAG_IDS[0], status, None, None, status.value, "ObserverFailure"
    )
    observations[0] = replace(
        original, scores=(failed_score, original.scores[1])
    )
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    affected = next(
        item
        for item in assessment.bounds
        if item.tag_id == OPAQUE_TAG_IDS[0]
        and item.error_cluster_count == 1
    )
    assert affected.cluster_count == 14
    assert affected.conditional_error_upper_ppm == 377_292

    with pytest.raises(PrototypeSceneCalibrationError, match="all 28"):
        assess_prototype_scene_calibration(
            plan,
            observations[:-1],
            expected_calibration_plan_digest=plan.record_digest,
        )


def test_explicit_observer_indeterminate_is_an_abstention_not_an_error() -> None:
    _cohort_plan, plan = _calibration_plan()
    observations = list(_observations(plan))
    original = observations[0]
    abstain = PrototypeSceneTagScore(
        OPAQUE_TAG_IDS[0],
        PrototypeSceneScoreStatus.INDETERMINATE,
        None,
        None,
        "genuinely_unresolvable",
        "PrototypeSceneIndeterminate",
    )
    observations[0] = replace(original, scores=(abstain, original.scores[1]))
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    affected = next(
        item
        for item in assessment.bounds
        if item.tag_id == OPAQUE_TAG_IDS[0]
        and item.abstention_cluster_count == 1
    )
    assert affected.error_cluster_count == 0
    assert affected.coverage_gate_accepted is False
    assert affected.accepted is False


def test_between_thresholds_is_abstention_not_false_decision() -> None:
    _cohort_plan, plan = _calibration_plan()
    observations = list(_observations(plan))
    original = observations[0]
    abstain = PrototypeSceneTagScore(
        OPAQUE_TAG_IDS[0],
        PrototypeSceneScoreStatus.SCORE,
        400_000,
        600_000,
        "scored",
        None,
    )
    observations[0] = replace(original, scores=(abstain, original.scores[1]))
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    affected = next(
        item
        for item in assessment.bounds
        if item.tag_id == OPAQUE_TAG_IDS[0]
        and item.abstention_cluster_count == 1
    )
    assert affected.error_cluster_count == 0
    assert affected.coverage_gate_accepted is False
    assert affected.accepted is False
    assert assessment.all_four_coverage_gates_accepted is False
    assert assessment.all_four_bounds_accepted is False
    with pytest.raises(PrototypeSceneCalibrationError, match="zero-abstention"):
        fit_prototype_scene_calibration_family(
            plan,
            observations,
            expected_calibration_plan_digest=plan.record_digest,
        )


def test_all_abstentions_are_typed_but_cannot_certify_a_family() -> None:
    _cohort_plan, plan = _calibration_plan()
    observations = tuple(
        replace(
            observation,
            scores=tuple(
                PrototypeSceneTagScore(
                    tag_id,
                    PrototypeSceneScoreStatus.SCORE,
                    400_000,
                    600_000,
                    "scored",
                    None,
                )
                for tag_id in OPAQUE_TAG_IDS
            ),
        )
        for observation in _observations(plan)
    )
    assessment = assess_prototype_scene_calibration(
        plan,
        observations,
        expected_calibration_plan_digest=plan.record_digest,
    )
    assert assessment.all_four_coverage_gates_accepted is False
    assert assessment.all_four_bounds_accepted is False
    assert all(item.error_cluster_count == 0 for item in assessment.bounds)
    assert all(item.abstention_cluster_count == 14 for item in assessment.bounds)
    assert all(item.conditional_error_upper_ppm == 268_752 for item in assessment.bounds)
    assert not any(item.coverage_gate_accepted for item in assessment.bounds)
    assert not any(item.accepted for item in assessment.bounds)
    with pytest.raises(PrototypeSceneCalibrationError, match="zero-abstention"):
        fit_prototype_scene_calibration_family(
            plan,
            observations,
            expected_calibration_plan_digest=plan.record_digest,
        )


def test_threshold_and_identity_drift_fail_closed() -> None:
    cohort, plan = _calibration_plan()
    with pytest.raises(PrototypeSceneCalibrationError, match="authority"):
        replace(plan, calibration_algorithm_digest="sha256:" + "0" * 64)
    with pytest.raises(PrototypeSceneCalibrationError, match="threshold commitment"):
        replace(
            plan,
            thresholds=(
                PrototypeSceneTagThreshold(OPAQUE_TAG_IDS[0], 200_000, 700_000),
                plan.thresholds[1],
            ),
        )
    with pytest.raises(PrototypeSceneCalibrationError, match="description catalog"):
        create_prototype_scene_calibration_plan(
            cohort_plan=cohort,
            thresholds=plan.thresholds,
            description_catalog_digest=_address("different"),
            prototype_reference_digest=plan.prototype_reference_digest,
            observer_protocol_id=plan.observer_protocol_id,
            observer_protocol_digest=plan.observer_protocol_digest,
            model_id=plan.model_id,
            model_identity_digest=plan.model_identity_digest,
            environment_digest=plan.environment_digest,
            expected_cohort_plan_digest=cohort.record_digest,
            expected_threshold_commitment=plan.threshold_commitment,
            expected_description_catalog_digest=plan.description_catalog_digest,
            expected_prototype_reference_digest=plan.prototype_reference_digest,
            expected_observer_protocol_digest=plan.observer_protocol_digest,
            expected_model_identity_digest=plan.model_identity_digest,
            expected_environment_digest=plan.environment_digest,
        )
    observations = list(_observations(plan))
    observations[0] = replace(
        observations[0], prototype_reference_digest=_address("drift")
    )
    with pytest.raises(PrototypeSceneCalibrationError, match="drift"):
        assess_prototype_scene_calibration(
            plan,
            observations,
            expected_calibration_plan_digest=plan.record_digest,
        )

    changed_cohort = replace(cohort, namespace="changed-cohort-namespace")
    with pytest.raises(PrototypeSceneCalibrationError):
        verify_prototype_scene_calibration_family(
            fit_prototype_scene_calibration_family(
                plan,
                _observations(plan),
                expected_calibration_plan_digest=plan.record_digest,
            ),
            calibration_plan=plan,
            cohort_plan=changed_cohort,
            observations=_observations(plan),
            expected_family_digest=fit_prototype_scene_calibration_family(
                plan,
                _observations(plan),
                expected_calibration_plan_digest=plan.record_digest,
            ).record_digest,
            expected_calibration_plan_digest=plan.record_digest,
            expected_cohort_plan_digest=cohort.record_digest,
        )


def test_dynamic_evaluator_gates_identity_population_and_score_state() -> None:
    _cohort_plan, plan = _calibration_plan()
    family = fit_prototype_scene_calibration_family(
        plan,
        _observations(plan),
        expected_calibration_plan_digest=plan.record_digest,
    )
    context = PrototypeSceneEvaluationContext(
        cohort_plan_digest=family.cohort_plan_digest,
        description_catalog_digest=family.description_catalog_digest,
        prototype_reference_digest=family.prototype_reference_digest,
        observer_protocol_id=family.observer_protocol_id,
        observer_protocol_digest=family.observer_protocol_digest,
        model_id=family.model_id,
        model_identity_digest=family.model_identity_digest,
        environment_digest=family.environment_digest,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )
    present = _score(OPAQUE_TAG_IDS[0], "present")
    result = evaluate_prototype_scene_score(family, present, context)
    assert result.disposition is PrototypeSceneDisposition.CALIBRATED_PRESENT
    assert result.identity_valid is True
    assert type(result).from_data(result.to_data()) == result

    drift = replace(context, environment_digest=_address("other-environment"))
    assert evaluate_prototype_scene_score(
        family, present, drift
    ).disposition is PrototypeSceneDisposition.ERROR
    invalid_population = replace(
        context, same_basic_renderer_population_valid=False
    )
    assert evaluate_prototype_scene_score(
        family, present, invalid_population
    ).disposition is PrototypeSceneDisposition.ERROR


def test_no_lean_dependency_and_python_authority_is_explicit() -> None:
    _cohort_plan, plan = _calibration_plan()
    family = fit_prototype_scene_calibration_family(
        plan,
        _observations(plan),
        expected_calibration_plan_digest=plan.record_digest,
    )
    assert plan.lean_required is False
    assert plan.lean_defines_identity_or_decision is False
    assert plan.lean_required_for_replay is False
    assert family.lean_required is False
    assert family.lean_defines_identity_or_decision is False
    assert family.lean_required_for_replay is False
    module = inspect.getmodule(create_prototype_scene_calibration_plan)
    assert module is not None
    imported = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("lean" in name.lower() for name in imported)
