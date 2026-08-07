from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

import bongard.prototype_scene_runtime_adapter as runtime_adapter_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS, plan_prototype_pair_cohort
from bongard.prototype_scene_calibration import (
    OBSERVER_ADAPTER_PROTOCOL_ID,
    PrototypeSceneCalibrationObservation,
    PrototypeSceneDisposition,
    PrototypeSceneEvaluationContext,
    PrototypeSceneScoreStatus,
    PrototypeSceneTagScore,
    PrototypeSceneTagThreshold,
    create_prototype_scene_calibration_plan,
    fit_prototype_scene_calibration_family,
    threshold_commitment,
)
from bongard.prototype_scene_observer import (
    PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
    PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
    PrototypeSceneObserverStatus,
    build_prototype_reference_catalog,
    describe_prototype_references,
    observe_prototype_scene,
    prototype_scene_observer_model_digest,
    prototype_scene_scoring_protocol_digest,
)
from bongard.prototype_scene_headless_runner import (
    PrototypeSceneCandidateFreeze,
    PrototypeSceneFreezeCommitReceipt,
    prototype_scene_rank_input_digest,
)
from bongard.prototype_scene_predicates import (
    PrototypeScenePredicateLibrary,
    PrototypeSceneVerifiedObserverBinding,
)
from bongard.prototype_scene_runtime_adapter import (
    PrototypeSceneArtifactPurpose,
    PrototypeScenePhasedArtifactVerifier,
    PrototypeSceneRuntimeAdapterError,
    PrototypeSceneRuntimeArtifactArchive,
    PrototypeSceneRuntimeArtifactInput,
    materialize_prototype_scene_calibration_observation,
    materialize_prototype_scene_calibration_observations,
    materialize_prototype_scene_panel,
    prototype_scene_evaluation_context_digest,
    prototype_scene_runtime_adapter_source_digest,
)
from bongard.prototype_scene_support_version_space import (
    build_prototype_scene_support_version_space,
    complete_prototype_scene_candidates,
    rank_prototype_scene_survivors,
)
from bongard.tests.test_prototype_pair_cohort import _fixture, _kwargs
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    _description_payload,
    _png,
    _receipt,
)
from bongard.tests.test_prototype_scene_headless_pipeline import _rank_response
from bongard.transport import CodexStructuredResult


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _calibration_score(tag_id: str, expected: str) -> PrototypeSceneTagScore:
    value = 1_000_000 if expected == "present" else 0
    return PrototypeSceneTagScore(
        tag_id=tag_id,
        status=PrototypeSceneScoreStatus.SCORE,
        lower_ppm=value,
        upper_ppm=value,
        reason_code="scored",
        error_type=None,
    )


@pytest.fixture(scope="module")
def runtime_authority():
    historical, release, split, inventory, exposure, _candidate_ids = _fixture()
    cohort = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )
    reference_ids = tuple(
        panel_id
        for prototype in cohort.prototypes
        for panel_id in prototype.panel_ids
    )
    references = {
        panel_id: _png(index) for index, panel_id in enumerate(reference_ids)
    }
    reference_sha256 = {
        panel_id: hashlib.sha256(payload).hexdigest()
        for panel_id, payload in references.items()
    }
    catalog = build_prototype_reference_catalog(
        cohort,
        references,
        expected_plan_digest=cohort.record_digest,
        expected_reference_sha256=reference_sha256,
    )

    def description_transport(prompt, paths, names, schema, **_kwargs):
        payload = _description_payload()
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    rubric = describe_prototype_references(
        catalog,
        references,
        expected_catalog_digest=catalog.catalog_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        transport=description_transport,
    )
    thresholds = tuple(
        PrototypeSceneTagThreshold(tag_id, 300_000, 700_000)
        for tag_id in OPAQUE_TAG_IDS
    )
    description_digest = "sha256:" + rubric.artifact_digest
    reference_digest = "sha256:" + catalog.catalog_digest
    protocol_digest = "sha256:" + prototype_scene_scoring_protocol_digest()
    model_digest = "sha256:" + prototype_scene_observer_model_digest(
        MODEL, EFFORT
    )
    environment_digest = "sha256:" + rubric.environment_digest
    calibration_plan = create_prototype_scene_calibration_plan(
        cohort_plan=cohort,
        thresholds=thresholds,
        description_catalog_digest=description_digest,
        prototype_reference_digest=reference_digest,
        observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
        observer_protocol_digest=protocol_digest,
        model_id=MODEL,
        model_identity_digest=model_digest,
        environment_digest=environment_digest,
        expected_cohort_plan_digest=cohort.record_digest,
        expected_threshold_commitment=threshold_commitment(thresholds),
        expected_description_catalog_digest=description_digest,
        expected_prototype_reference_digest=reference_digest,
        expected_observer_protocol_digest=protocol_digest,
        expected_model_identity_digest=model_digest,
        expected_environment_digest=environment_digest,
    )
    observations = []
    for scheduled in calibration_plan.scenes:
        expected = dict(scheduled.expected_tag_states)
        observations.append(
            PrototypeSceneCalibrationObservation(
                calibration_plan_digest=calibration_plan.record_digest,
                cohort_plan_digest=calibration_plan.cohort_plan_digest,
                task_id=scheduled.task_id,
                panel_id=scheduled.panel_id,
                observer_artifact_digest=_address(
                    {"task": scheduled.task_id, "panel": scheduled.panel_id}
                ),
                observer_artifact_schema=PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
                description_catalog_digest=description_digest,
                prototype_reference_digest=reference_digest,
                observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
                observer_protocol_digest=protocol_digest,
                model_id=MODEL,
                model_identity_digest=model_digest,
                environment_digest=environment_digest,
                observer_call_count=1,
                scores=tuple(
                    _calibration_score(tag_id, expected[tag_id])
                    for tag_id in OPAQUE_TAG_IDS
                ),
                adapter_protocol_id=OBSERVER_ADAPTER_PROTOCOL_ID,
            )
        )
    family = fit_prototype_scene_calibration_family(
        calibration_plan,
        observations,
        expected_calibration_plan_digest=calibration_plan.record_digest,
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
    return (
        cohort,
        references,
        reference_sha256,
        catalog,
        rubric,
        calibration_plan,
        family,
        context,
    )


def _observer_artifact(
    runtime_authority,
    *,
    transport_failure: bool = False,
    parser_failure: bool = False,
):
    (
        cohort,
        references,
        _reference_sha256,
        catalog,
        rubric,
        _calibration_plan,
        _family,
        context,
    ) = runtime_authority
    scene = _png(31 if not transport_failure and not parser_failure else 32)
    panel_id = (
        cohort.drill.positive_panel_ids[0]
        if not transport_failure and not parser_failure
        else cohort.drill.negative_panel_ids[0]
    )

    def transport(prompt, paths, names, schema, **_kwargs):
        if transport_failure:
            raise RuntimeError("offline fixture transport failure")
        payload = {
            "description": "One angular bird-like object is visibly present.",
            "cells": [
                {
                    "group_id": "group_0",
                    "state": "scored",
                    "lower_ppm": 1_000_000,
                    "upper_ppm": 1_000_000,
                    "reason_code": None,
                },
                {
                    "group_id": "group_1",
                    "state": "indeterminate",
                    "lower_ppm": None,
                    "upper_ppm": None,
                    "reason_code": "genuinely_unresolvable",
                },
            ],
        }
        if parser_failure:
            payload["cells"].pop()
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_prototype_scene(
        scene,
        scene_task_id=cohort.drill.task_id,
        scene_panel_id=panel_id,
        observation_context_digest=prototype_scene_evaluation_context_digest(
            context
        ),
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        catalog=catalog,
        prototype_png_by_panel_id=references,
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact=rubric,
        expected_rubric_artifact_digest=rubric.artifact_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        transport=transport,
    )
    return scene, artifact


def _observe_scheduled_panel(
    runtime_authority,
    *,
    task_id: str,
    panel_id: str,
    scene_seed: int,
    observation_context_digest: str,
    states: tuple[str, str],
):
    (
        _cohort,
        references,
        _reference_sha256,
        catalog,
        rubric,
        _calibration_plan,
        _family,
        _context,
    ) = runtime_authority
    scene = _png(scene_seed)

    def transport(prompt, paths, names, schema, **_kwargs):
        cells = []
        for index, state in enumerate(states):
            if state == "indeterminate":
                cells.append(
                    {
                        "group_id": f"group_{index}",
                        "state": "indeterminate",
                        "lower_ppm": None,
                        "upper_ppm": None,
                        "reason_code": "genuinely_unresolvable",
                    }
                )
            else:
                value = 1_000_000 if state == "present" else 0
                cells.append(
                    {
                        "group_id": f"group_{index}",
                        "state": "scored",
                        "lower_ppm": value,
                        "upper_ppm": value,
                        "reason_code": None,
                    }
                )
        payload = {
            "description": "One bounded scene observation for offline replay.",
            "cells": cells,
        }
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_prototype_scene(
        scene,
        scene_task_id=task_id,
        scene_panel_id=panel_id,
        observation_context_digest=observation_context_digest,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        catalog=catalog,
        prototype_png_by_panel_id=references,
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact=rubric,
        expected_rubric_artifact_digest=rubric.artifact_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        transport=transport,
    )
    return scene, artifact


def _seal_archive_entries(runtime_authority, entries):
    (
        _cohort,
        references,
        reference_sha256,
        catalog,
        rubric,
        _calibration_plan,
        _family,
        _context,
    ) = runtime_authority
    catalog_bytes = canonical_json(catalog.to_data())
    rubric_bytes = canonical_json(rubric.to_data())
    return PrototypeSceneRuntimeArtifactArchive.seal_external(
        archive_source_id="fixture.external.immutable-archive.v1",
        verifier_id="prototype.scene.archive-cold-verifier.v1",
        catalog_json_bytes=catalog_bytes,
        expected_catalog_json_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact_json_bytes=rubric_bytes,
        expected_rubric_artifact_json_sha256=hashlib.sha256(
            rubric_bytes
        ).hexdigest(),
        expected_rubric_artifact_digest=rubric.artifact_digest,
        prototype_reference_png_by_panel_id=references,
        expected_reference_sha256=reference_sha256,
        scenes=tuple(entries),
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )


def _archive(runtime_authority, scene, artifact):
    return _seal_archive_entries(
        runtime_authority, (_artifact_input(scene, artifact),)
    )


def _artifact_input(
    scene: bytes,
    artifact,
    *,
    purpose: PrototypeSceneArtifactPurpose = (
        PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION
    ),
) -> PrototypeSceneRuntimeArtifactInput:
    artifact_bytes = canonical_json(artifact.to_data())
    return PrototypeSceneRuntimeArtifactInput(
        scene_task_id=artifact.scene_task_id,
        panel_id=artifact.scene_panel_id,
        expected_observation_context_digest=artifact.observation_context_digest,
        exact_scene_png_bytes=scene,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        observer_artifact_json_bytes=artifact_bytes,
        expected_observer_artifact_json_sha256=hashlib.sha256(
            artifact_bytes
        ).hexdigest(),
        expected_observer_artifact_digest=artifact.artifact_digest,
        purpose=purpose,
    )


def _phased_archives_and_freeze(runtime_authority):
    cohort = runtime_authority[0]
    family = runtime_authority[-2]
    context_digest = prototype_scene_evaluation_context_digest(
        runtime_authority[-1]
    )
    support_entries = []
    for index, panel_id in enumerate(cohort.drill.positive_panel_ids[:6]):
        scene, artifact = _observe_scheduled_panel(
            runtime_authority,
            task_id=cohort.drill.task_id,
            panel_id=panel_id,
            scene_seed=10 + index,
            observation_context_digest=context_digest,
            states=("present", "present"),
        )
        support_entries.append(_artifact_input(scene, artifact))
    for index, panel_id in enumerate(cohort.drill.negative_panel_ids[:6]):
        scene, artifact = _observe_scheduled_panel(
            runtime_authority,
            task_id=cohort.drill.task_id,
            panel_id=panel_id,
            scene_seed=20 + index,
            observation_context_digest=context_digest,
            states=(
                ("absent", "present")
                if index % 2 == 0
                else ("present", "absent")
            ),
        )
        support_entries.append(_artifact_input(scene, artifact))
    support_archive = _seal_archive_entries(
        runtime_authority, support_entries
    )
    support_panels = tuple(
        materialize_prototype_scene_panel(
            support_archive,
            family,
            entry.panel_id,
            expected_archive_digest=support_archive.record_digest,
        )
        for entry in support_entries
    )
    library = PrototypeScenePredicateLibrary.freeze(family)
    version = build_prototype_scene_support_version_space(
        library, family, support_panels[:6], support_panels[6:]
    )
    rank_input = prototype_scene_rank_input_digest(
        library_digest=library.record_digest,
        version_space_digest=version.record_digest,
        survivor_candidate_ids=version.survivor_candidate_ids,
    )
    response = _rank_response(version.survivor_candidate_ids, rank_input)
    ranking = rank_prototype_scene_survivors(
        version, library, response.ordered_candidate_ids
    )
    selected = next(
        item
        for item in complete_prototype_scene_candidates(library)
        if item.candidate_id == ranking.ordered_candidate_ids[0]
    )
    freeze = PrototypeSceneCandidateFreeze.seal(
        library=library,
        family=family,
        support_digest=_address(
            {
                "schema": "gkm.bongard-prototype-scene-headless-support.v1",
                "panels": [item.to_data() for item in support_panels],
                "sides": ["positive"] * 6 + ["negative"] * 6,
                **runtime_adapter_module._authority_data(),  # noqa: SLF001
            }
        ),
        version=version,
        ranking=ranking,
        rank_response=response,
        selected_candidate=selected,
    )
    freeze_bytes = canonical_json(freeze.to_data()) + b"\n"
    freeze_commit = PrototypeSceneFreezeCommitReceipt.seal(
        freeze,
        freeze_bytes,
        storage_id="fixture-durable-candidate-freeze",
    )

    query_entries = []
    for index, (panel_id, states) in enumerate(
        (
            (
                cohort.drill.positive_panel_ids[6],
                ("present", "present"),
            ),
            (
                cohort.drill.negative_panel_ids[6],
                ("absent", "present"),
            ),
        )
    ):
        scene, artifact = _observe_scheduled_panel(
            runtime_authority,
            task_id=cohort.drill.task_id,
            panel_id=panel_id,
            scene_seed=30 + index,
            observation_context_digest=context_digest,
            states=states,
        )
        query_entries.append(_artifact_input(scene, artifact))
    query_archive = _seal_archive_entries(runtime_authority, query_entries)
    query_panels = tuple(
        materialize_prototype_scene_panel(
            query_archive,
            family,
            entry.panel_id,
            expected_archive_digest=query_archive.record_digest,
        )
        for entry in query_entries
    )
    return (
        support_archive,
        support_entries,
        support_panels,
        query_archive,
        query_entries,
        query_panels,
        freeze,
        freeze_commit,
    )


def test_materialization_replays_raw_archive_and_preserves_indeterminate(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority)
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    archive = _archive(runtime_authority, scene, artifact)
    family = runtime_authority[-2]
    panel = materialize_prototype_scene_panel(
        archive,
        family,
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )

    assert tuple(item.status for item in panel.scores) == (
        PrototypeSceneScoreStatus.SCORE,
        PrototypeSceneScoreStatus.INDETERMINATE,
    )
    assert tuple(item.disposition for item in panel.results) == (
        PrototypeSceneDisposition.CALIBRATED_PRESENT,
        PrototypeSceneDisposition.INDETERMINATE,
    )
    assert panel.observer_binding.verifier_id == archive.verifier_id
    assert panel.observer_binding.verifier_digest == archive.verifier_digest
    assert panel.to_data()["python_is_canonical_authority"] is True
    assert panel.to_data()["lean_required"] is False
    archive.artifact_verifier(
        expected_archive_digest=archive.record_digest
    )(panel.observer_binding, panel.exact_png_bytes)


def test_archive_verifier_rejects_binding_self_attestation_and_wrong_anchor(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority)
    archive = _archive(runtime_authority, scene, artifact)
    family = runtime_authority[-2]
    context = runtime_authority[-1]
    panel = materialize_prototype_scene_panel(
        archive,
        family,
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )
    forged_scores = (
        PrototypeSceneTagScore(
            OPAQUE_TAG_IDS[0],
            PrototypeSceneScoreStatus.SCORE,
            0,
            0,
            "scored",
            None,
        ),
        panel.scores[1],
    )
    forged = PrototypeSceneVerifiedObserverBinding.seal_verified(
        panel_id=panel.panel_id,
        exact_png_bytes=scene,
        observer_artifact_schema=panel.observer_binding.observer_artifact_schema,
        observer_artifact_digest=panel.observer_binding.observer_artifact_digest,
        verifier_id=archive.verifier_id,
        verifier_digest=archive.verifier_digest,
        scores=forged_scores,
        context=context,
    )
    verifier = archive.artifact_verifier(
        expected_archive_digest=archive.record_digest
    )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="archive reconstruction"
    ):
        verifier(forged, scene)
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="external commitment"
    ):
        archive.artifact_verifier(expected_archive_digest="sha256:" + "0" * 64)


def test_transport_failure_is_typed_error_never_absence(runtime_authority) -> None:
    scene, artifact = _observer_artifact(
        runtime_authority, transport_failure=True
    )
    assert artifact.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR
    archive = _archive(runtime_authority, scene, artifact)
    panel = materialize_prototype_scene_panel(
        archive,
        runtime_authority[-2],
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )
    assert all(
        item.status is PrototypeSceneScoreStatus.TRANSPORT_ERROR
        for item in panel.scores
    )
    assert all(
        item.disposition is PrototypeSceneDisposition.ERROR
        for item in panel.results
    )
    assert all(item.lower_ppm is None and item.upper_ppm is None for item in panel.scores)


def test_raw_byte_commitments_and_context_precommit_fail_closed(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority)
    raw = canonical_json(artifact.to_data())
    good = PrototypeSceneRuntimeArtifactInput(
        scene_task_id=artifact.scene_task_id,
        panel_id=artifact.scene_panel_id,
        expected_observation_context_digest=artifact.observation_context_digest,
        exact_scene_png_bytes=scene,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        observer_artifact_json_bytes=raw,
        expected_observer_artifact_json_sha256=hashlib.sha256(raw).hexdigest(),
        expected_observer_artifact_digest=artifact.artifact_digest,
    )
    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="byte commitment"):
        replace(good, observer_artifact_json_bytes=raw + b"\n")
    wrong_context = replace(
        good, expected_observation_context_digest="sha256:" + "e" * 64
    )
    (
        _cohort,
        references,
        reference_sha256,
        catalog,
        rubric,
        _calibration_plan,
        _family,
        _context,
    ) = runtime_authority
    catalog_bytes = canonical_json(catalog.to_data())
    rubric_bytes = canonical_json(rubric.to_data())
    with pytest.raises(Exception, match="parent reconstruction|context"):
        PrototypeSceneRuntimeArtifactArchive.seal_external(
            archive_source_id="fixture.external.immutable-archive.v1",
            verifier_id="prototype.scene.archive-cold-verifier.v1",
            catalog_json_bytes=catalog_bytes,
            expected_catalog_json_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            expected_catalog_digest=catalog.catalog_digest,
            rubric_artifact_json_bytes=rubric_bytes,
            expected_rubric_artifact_json_sha256=hashlib.sha256(
                rubric_bytes
            ).hexdigest(),
            expected_rubric_artifact_digest=rubric.artifact_digest,
            prototype_reference_png_by_panel_id=references,
            expected_reference_sha256=reference_sha256,
            scenes=(wrong_context,),
            same_basic_renderer_population_valid=True,
            conditional_transport_assumption_accepted=True,
            observer_environment_valid=True,
        )


def test_stale_rehashed_observer_artifact_and_archive_mutation_are_rejected(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority)
    stale = deepcopy(artifact.to_data())
    stale["source_digest"] = "0" * 64
    stale_bytes = canonical_json(stale)
    entry = PrototypeSceneRuntimeArtifactInput(
        scene_task_id=artifact.scene_task_id,
        panel_id=artifact.scene_panel_id,
        expected_observation_context_digest=artifact.observation_context_digest,
        exact_scene_png_bytes=scene,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        observer_artifact_json_bytes=stale_bytes,
        # The attacker may rehash changed raw bytes; the typed artifact and its
        # independently committed logical digest must still reject them.
        expected_observer_artifact_json_sha256=hashlib.sha256(
            stale_bytes
        ).hexdigest(),
        expected_observer_artifact_digest=artifact.artifact_digest,
    )
    with pytest.raises(Exception, match="source"):
        _seal_archive_entries(runtime_authority, (entry,))

    archive = _archive(runtime_authority, scene, artifact)
    panel = materialize_prototype_scene_panel(
        archive,
        runtime_authority[-2],
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )
    object.__setattr__(
        archive, "catalog_json_bytes", archive.catalog_json_bytes + b"\n"
    )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="byte commitment|changed"
    ):
        archive.artifact_verifier(
            expected_archive_digest=archive.record_digest
        )(panel.observer_binding, panel.exact_png_bytes)


def test_wrong_panel_png_verifier_identity_and_family_are_rejected(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority)
    archive = _archive(runtime_authority, scene, artifact)
    family = runtime_authority[-2]
    context = runtime_authority[-1]
    panel = materialize_prototype_scene_panel(
        archive,
        family,
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )
    verifier = archive.artifact_verifier(
        expected_archive_digest=archive.record_digest
    )

    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="runner PNG"):
        verifier(panel.observer_binding, _png(99))

    unknown_panel = PrototypeSceneVerifiedObserverBinding.seal_verified(
        panel_id="bd/unknown-runtime-panel/0/0.png",
        exact_png_bytes=scene,
        observer_artifact_schema=panel.observer_binding.observer_artifact_schema,
        observer_artifact_digest=panel.observer_binding.observer_artifact_digest,
        verifier_id=archive.verifier_id,
        verifier_digest=archive.verifier_digest,
        scores=panel.scores,
        context=context,
    )
    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="absent"):
        verifier(unknown_panel, scene)

    stale_verifier = PrototypeSceneVerifiedObserverBinding.seal_verified(
        panel_id=panel.panel_id,
        exact_png_bytes=scene,
        observer_artifact_schema=panel.observer_binding.observer_artifact_schema,
        observer_artifact_digest=panel.observer_binding.observer_artifact_digest,
        verifier_id=archive.verifier_id,
        verifier_digest=_address("stale-runtime-verifier"),
        scores=panel.scores,
        context=context,
    )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="archive reconstruction"
    ):
        verifier(stale_verifier, scene)

    foreign_family = replace(family, model_id=family.model_id + "-stale")
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="context differs"
    ):
        materialize_prototype_scene_panel(
            archive,
            foreign_family,
            artifact.scene_panel_id,
            expected_archive_digest=archive.record_digest,
        )


def test_parser_failure_is_error_and_never_numerical_absence(
    runtime_authority,
) -> None:
    scene, artifact = _observer_artifact(runtime_authority, parser_failure=True)
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    archive = _archive(runtime_authority, scene, artifact)
    panel = materialize_prototype_scene_panel(
        archive,
        runtime_authority[-2],
        artifact.scene_panel_id,
        expected_archive_digest=archive.record_digest,
    )
    assert all(
        item.status is PrototypeSceneScoreStatus.PARSER_ERROR
        for item in panel.scores
    )
    assert all(
        item.disposition is PrototypeSceneDisposition.ERROR
        for item in panel.results
    )
    assert all(
        item.lower_ppm is None and item.upper_ppm is None
        for item in panel.scores
    )


def test_loaded_adapter_source_identity_fails_on_disk_drift(monkeypatch) -> None:
    expected = prototype_scene_runtime_adapter_source_digest()
    assert expected.startswith("sha256:")

    class ChangedSourcePath:
        def __init__(self, _value: object) -> None:
            pass

        def read_bytes(self) -> bytes:
            return b"changed adapter source after import"

    monkeypatch.setattr(runtime_adapter_module, "Path", ChangedSourcePath)
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="changed after.*loaded"
    ):
        prototype_scene_runtime_adapter_source_digest()


def test_calibration_observation_is_adapted_only_after_cold_verification(
    runtime_authority,
) -> None:
    calibration_plan = runtime_authority[-3]
    scheduled = calibration_plan.scenes[0]
    scene, artifact = _observe_scheduled_panel(
        runtime_authority,
        task_id=scheduled.task_id,
        panel_id=scheduled.panel_id,
        scene_seed=71,
        observation_context_digest=calibration_plan.record_digest,
        states=("present", "indeterminate"),
    )
    archive = _seal_archive_entries(
        runtime_authority,
        (
            _artifact_input(
                scene,
                artifact,
                purpose=PrototypeSceneArtifactPurpose.CALIBRATION,
            ),
        ),
    )
    observation = materialize_prototype_scene_calibration_observation(
        archive,
        calibration_plan,
        scheduled.task_id,
        scheduled.panel_id,
        expected_archive_digest=archive.record_digest,
    )
    assert tuple(item.status for item in observation.scores) == (
        PrototypeSceneScoreStatus.SCORE,
        PrototypeSceneScoreStatus.INDETERMINATE,
    )
    assert observation.observer_artifact_digest == (
        "sha256:" + artifact.artifact_digest
    )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="calibration artifact"
    ):
        materialize_prototype_scene_panel(
            archive,
            runtime_authority[-2],
            scheduled.panel_id,
            expected_archive_digest=archive.record_digest,
        )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="role, task"
    ):
        materialize_prototype_scene_calibration_observation(
            archive,
            calibration_plan,
            scheduled.task_id + "-wrong",
            scheduled.panel_id,
            expected_archive_digest=archive.record_digest,
        )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="complete frozen schedule"
    ):
        materialize_prototype_scene_calibration_observations(
            archive,
            calibration_plan,
            expected_archive_digest=archive.record_digest,
        )


def test_complete_calibration_archive_is_cold_adapted_in_plan_order(
    runtime_authority,
) -> None:
    calibration_plan = runtime_authority[-3]
    entries = []
    for index, scheduled in enumerate(calibration_plan.scenes):
        expected = dict(scheduled.expected_tag_states)
        scene, artifact = _observe_scheduled_panel(
            runtime_authority,
            task_id=scheduled.task_id,
            panel_id=scheduled.panel_id,
            scene_seed=100 + index,
            observation_context_digest=calibration_plan.record_digest,
            states=tuple(expected[tag_id] for tag_id in OPAQUE_TAG_IDS),
        )
        entries.append(
            _artifact_input(
                scene,
                artifact,
                purpose=PrototypeSceneArtifactPurpose.CALIBRATION,
            )
        )
    archive = _seal_archive_entries(runtime_authority, entries)
    observations = materialize_prototype_scene_calibration_observations(
        archive,
        calibration_plan,
        expected_archive_digest=archive.record_digest,
    )
    assert len(observations) == len(calibration_plan.scenes) == 28
    assert tuple((item.task_id, item.panel_id) for item in observations) == tuple(
        (item.task_id, item.panel_id) for item in calibration_plan.scenes
    )
    assert all(
        score.status is PrototypeSceneScoreStatus.SCORE
        for observation in observations
        for score in observation.scores
    )


def test_phased_verifier_attaches_query_once_only_after_freeze_and_cold_replays(
    runtime_authority,
) -> None:
    (
        support_archive,
        support_entries,
        support_panels,
        query_archive,
        _query_entries,
        query_panels,
        freeze,
        freeze_commit,
    ) = _phased_archives_and_freeze(runtime_authority)
    phased = PrototypeScenePhasedArtifactVerifier.for_support(
        support_archive,
        expected_support_archive_digest=support_archive.record_digest,
        family=runtime_authority[-2],
        support_panels=support_panels,
    )
    phased(support_panels[0].observer_binding, support_panels[0].exact_png_bytes)
    assert phased.query_archive_attached is False
    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="not attached"):
        phased(query_panels[0].observer_binding, query_panels[0].exact_png_bytes)

    phased.attach_query_archive_after_freeze(
        query_archive,
        expected_query_archive_digest=query_archive.record_digest,
        freeze=freeze,
        freeze_commit=freeze_commit,
        expected_freeze_commit_digest=freeze_commit.record_digest,
    )
    assert phased.query_archive_attached is True
    phased(query_panels[0].observer_binding, query_panels[0].exact_png_bytes)
    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="one-shot"):
        phased.attach_query_archive_after_freeze(
            query_archive,
            expected_query_archive_digest=query_archive.record_digest,
            freeze=freeze,
            freeze_commit=freeze_commit,
            expected_freeze_commit_digest=freeze_commit.record_digest,
        )

    cold = PrototypeScenePhasedArtifactVerifier.from_pinned_archives_for_cold_replay(
        support_archive,
        query_archive,
        expected_support_archive_digest=support_archive.record_digest,
        expected_query_archive_digest=query_archive.record_digest,
        family=runtime_authority[-2],
        support_panels=support_panels,
        freeze=freeze,
        freeze_commit=freeze_commit,
        expected_freeze_commit_digest=freeze_commit.record_digest,
    )
    cold(support_panels[-1].observer_binding, support_panels[-1].exact_png_bytes)
    cold(query_panels[-1].observer_binding, query_panels[-1].exact_png_bytes)

    overlapping_query = _seal_archive_entries(
        runtime_authority, support_entries[:2]
    )
    fresh = PrototypeScenePhasedArtifactVerifier.for_support(
        support_archive,
        expected_support_archive_digest=support_archive.record_digest,
        family=runtime_authority[-2],
        support_panels=support_panels,
    )
    with pytest.raises(PrototypeSceneRuntimeAdapterError, match="overlaps"):
        fresh.attach_query_archive_after_freeze(
            overlapping_query,
            expected_query_archive_digest=overlapping_query.record_digest,
            freeze=freeze,
            freeze_commit=freeze_commit,
            expected_freeze_commit_digest=freeze_commit.record_digest,
        )
    with pytest.raises(
        PrototypeSceneRuntimeAdapterError, match="freeze receipt.*commitment"
    ):
        PrototypeScenePhasedArtifactVerifier.for_support(
            support_archive,
            expected_support_archive_digest=support_archive.record_digest,
            family=runtime_authority[-2],
            support_panels=support_panels,
        ).attach_query_archive_after_freeze(
            query_archive,
            expected_query_archive_digest=query_archive.record_digest,
            freeze=freeze,
            freeze_commit=freeze_commit,
            expected_freeze_commit_digest="sha256:" + "0" * 64,
        )
