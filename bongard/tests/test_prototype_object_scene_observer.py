from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from bongard.prototype_object_hypotheses import extract_object_hypotheses
from bongard.prototype_object_profiles import OBJECT_FEATURE_IDS
from bongard.prototype_pair_cohort import plan_prototype_pair_cohort
from bongard.prototype_scene_calibration import adapt_prototype_scene_observation
from bongard.prototype_object_scene_observer import (
    PrototypeRubricDescriptionArtifact,
    PrototypeSceneObserverArtifact,
    PrototypeSceneObserverError,
    PrototypeSceneObserverStatus,
    PrototypeSceneScoreState,
    build_prototype_reference_catalog,
    describe_prototype_references,
    observe_prototype_scene,
    verify_prototype_rubric_description_artifact,
    verify_prototype_scene_observer_artifact,
)
from bongard.tests.test_prototype_pair_cohort import _fixture, _kwargs
from bongard.tests.test_prototype_scene_observer import (
    CONTEXT_DIGEST,
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


@pytest.fixture(scope="module")
def observer_inputs():
    historical, release, split, inventory, exposure, _ = _fixture()
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )
    panel_ids = tuple(
        panel_id for binding in plan.prototypes for panel_id in binding.panel_ids
    )
    references = {panel_id: _png(index) for index, panel_id in enumerate(panel_ids)}
    catalog = build_prototype_reference_catalog(
        plan,
        references,
        expected_plan_digest=plan.record_digest,
        expected_reference_sha256={
            panel_id: hashlib.sha256(data).hexdigest()
            for panel_id, data in references.items()
        },
    )
    return plan, references, catalog, _png(20)


def _description_payload() -> dict[str, object]:
    return {
        "profiles": [
            {
                "group_id": "group_0",
                "rubric": "An angular flying shape with two oblique wings.",
                "atoms": [
                    {
                        "feature_id": "bird_like_support_ppm",
                        "operator": "at_least",
                        "target": 500_000,
                    }
                ],
            },
            {
                "group_id": "group_1",
                "rubric": "A compact outline with at least one straight span.",
                "atoms": [
                    {
                        "feature_id": "straight_span_count",
                        "operator": "at_least",
                        "target": 1,
                    }
                ],
            },
        ]
    }


def _describe(references, catalog):
    payload = _description_payload()

    def transport(prompt, paths, names, schema, **_kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    return describe_prototype_references(
        catalog,
        references,
        expected_catalog_digest=catalog.catalog_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )


def _feature_payload(scene: bytes) -> dict[str, object]:
    packet = extract_object_hypotheses(scene)
    first_by_scenario = {
        scenario.scenario_id: scenario.hypotheses[0].hypothesis_id
        for scenario in packet.scenarios
        if scenario.hypotheses
    }
    cells: list[dict[str, object]] = []
    for sheet in packet.atlas_sheets:
        for slot in sheet.slots:
            for feature_id in OBJECT_FEATURE_IDS:
                value = 0
                if (
                    feature_id == "bird_like_support_ppm"
                    and first_by_scenario.get(slot.scenario_id) == slot.hypothesis_id
                ):
                    value = 900_000
                cells.append(
                    {
                        "slot_id": slot.slot_id,
                        "feature_id": feature_id,
                        "state": "scored",
                        "lower": value,
                        "upper": value,
                        "reason_code": None,
                    }
                )
    return {"description": "An angular drawing with two visible wings.", "cells": cells}


def test_profile_blind_scene_turn_evaluates_and_cold_replays(observer_inputs) -> None:
    plan, references, catalog, scene = observer_inputs
    description = _describe(references, catalog)
    assert description.status is PrototypeSceneObserverStatus.SUCCESS
    assert tuple(item.profile_id for item in description.profiles) == (
        "group_0",
        "group_1",
    )
    assert PrototypeRubricDescriptionArtifact.from_data(
        description.to_data(), expected_artifact_digest=description.artifact_digest
    ) == description
    verify_prototype_rubric_description_artifact(
        description,
        catalog,
        references,
        expected_catalog_digest=catalog.catalog_digest,
        expected_artifact_digest=description.artifact_digest,
    )

    payload = _feature_payload(scene)
    seen: dict[str, object] = {}

    def transport(prompt, paths, names, schema, **_kwargs):
        seen["prompt"] = prompt
        seen["names"] = tuple(names)
        assert all(name.startswith("sheet_") for name in names)
        assert "group_0_ref" not in prompt and "group_1_ref" not in prompt
        assert "An angular flying shape" not in prompt
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    scene_task_id = plan.drill.task_id
    scene_panel_id = plan.drill.positive_panel_ids[0]
    artifact = observe_prototype_scene(
        scene,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        observation_context_digest=CONTEXT_DIGEST,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        catalog=catalog,
        prototype_png_by_panel_id=references,
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact=description,
        expected_rubric_artifact_digest=description.artifact_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert seen["names"]
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.hypothesis_packet is not None
    assert len(artifact.local_packets) == 3
    assert len(artifact.evaluations) == 2
    assert tuple((x.state, x.lower_ppm, x.upper_ppm) for x in artifact.scores) == (
        (PrototypeSceneScoreState.SCORED, 1_000_000, 1_000_000),
        (PrototypeSceneScoreState.SCORED, 0, 0),
    )
    assert PrototypeSceneObserverArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=artifact.artifact_digest
    ) == artifact
    assert verify_prototype_scene_observer_artifact(
        artifact,
        scene,
        expected_scene_task_id=scene_task_id,
        expected_scene_panel_id=scene_panel_id,
        expected_observation_context_digest=CONTEXT_DIGEST,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        catalog=catalog,
        prototype_png_by_panel_id=references,
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact=description,
        expected_rubric_artifact_digest=description.artifact_digest,
        expected_artifact_digest=artifact.artifact_digest,
    ) is artifact
    adapted = adapt_prototype_scene_observation(
        artifact, calibration_plan_digest=CONTEXT_DIGEST
    )
    assert tuple((item.lower_ppm, item.upper_ppm) for item in adapted.scores) == (
        (1_000_000, 1_000_000),
        (0, 0),
    )


def test_incomplete_feature_payload_is_error_not_absence(observer_inputs) -> None:
    plan, references, catalog, scene = observer_inputs
    description = _describe(references, catalog)
    payload = _feature_payload(scene)
    payload["cells"] = payload["cells"][:-1]  # type: ignore[index]

    def transport(prompt, paths, names, schema, **_kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_prototype_scene(
        scene,
        scene_task_id=plan.drill.task_id,
        scene_panel_id=plan.drill.positive_panel_ids[0],
        observation_context_digest=CONTEXT_DIGEST,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        catalog=catalog,
        prototype_png_by_panel_id=references,
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact=description,
        expected_rubric_artifact_digest=description.artifact_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.hypothesis_packet is not None
    assert artifact.local_packets == () and artifact.evaluations == ()
    assert all(item.state is PrototypeSceneScoreState.ERROR for item in artifact.scores)


def test_nested_profile_or_scene_tamper_is_rejected(observer_inputs) -> None:
    _plan, references, catalog, _scene = observer_inputs
    description = _describe(references, catalog)
    tampered = deepcopy(description.to_data())
    tampered["profiles"][0]["atoms"][0]["target"] = 1  # type: ignore[index]
    with pytest.raises((PrototypeSceneObserverError, ValueError)):
        PrototypeRubricDescriptionArtifact.from_data(tampered)
