"""Offline tests for prose-grounded ordinal rubric observations."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricObserverError,
    ObjectBongardRubricSpec,
    RUBRIC_ORDINAL_LEVEL_ANCHORS,
    RubricObservationState,
    object_bongard_rubric_observer_prompt,
    object_bongard_rubric_ordinal_scale_digest,
    observe_object_bongard_rubric,
    verify_object_bongard_rubric_observer_artifact,
)
from bongard.prototype_object_hypotheses import extract_object_hypothesis_packet
from bongard.prototype_object_lineages import extract_object_lineage_packet
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


PANEL_ID = "bd/bd_rubric_fixture_0000/0/0.png"
SEMANTIC_DIGEST = "a" * 64


def _inputs():
    scene = _png(27)
    hypotheses = extract_object_hypothesis_packet(scene)
    lineages = extract_object_lineage_packet(scene, hypotheses)
    spec = ObjectBongardRubricSpec.create(
        SEMANTIC_DIGEST,
        "A rounded bird-like contour arrangement recurs.",
        ("bird_like_support_ppm",),
    )
    return scene, hypotheses, lineages, spec


def _payload(sheet, *, lower: int = 1, upper: int = 2):
    return {
        "scene": {"lower": 2, "upper": 3},
        "slots": [
            {"slot_id": slot.slot_id, "lower": lower, "upper": upper}
            for slot in sheet.slots
        ],
    }


def _observe(*, fail: bool = False):
    scene, hypotheses, lineages, spec = _inputs()
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        sheet = hypotheses.atlas_sheets[calls]
        calls += 1
        assert names == ("scene.png", sheet.name)
        assert spec.rubric in prompt
        assert tuple(level for level, _ in RUBRIC_ORDINAL_LEVEL_ANCHORS) == tuple(
            range(5)
        )
        if fail:
            raise RuntimeError("synthetic transport failure")
        payload = _payload(sheet)
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_object_bongard_rubric(
        scene,
        panel_id=PANEL_ID,
        rubric_spec=spec,
        hypothesis_packet=hypotheses,
        lineage_packet=lineages,
        expected_scene_sha256=hashlib.sha256(scene).hexdigest(),
        expected_rubric_spec_digest=spec.spec_digest,
        expected_hypothesis_packet_digest=hypotheses.digest(),
        expected_lineage_packet_digest=lineages.digest(),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == len(hypotheses.atlas_sheets)
    return artifact, scene, hypotheses, lineages, spec


def test_live_ordinal_rows_project_and_cold_replay() -> None:
    artifact, scene, hypotheses, lineages, spec = _observe()
    assert artifact.physical_call_count == len(hypotheses.atlas_sheets)
    assert artifact.object_observations
    assert all(
        item.state is RubricObservationState.SCORED
        for item in artifact.object_observations
    )
    assert artifact.canonical_scene_observation is not None
    assert artifact.canonical_scene_observation.interval is not None
    assert artifact.canonical_scene_observation.interval.to_data() == {
        "lower": 2,
        "upper": 3,
    }
    assert ObjectBongardRubricObserverArtifact.from_data(
        artifact.to_data()
    ) == artifact
    assert verify_object_bongard_rubric_observer_artifact(
        artifact,
        scene,
        panel_id=PANEL_ID,
        rubric_spec=spec,
        hypothesis_packet=hypotheses,
        lineage_packet=lineages,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    assert spec == ObjectBongardRubricSpec.from_data(spec.to_data())
    assert len(object_bongard_rubric_ordinal_scale_digest()) == 64


def test_receipted_payload_cannot_be_decoupled_from_projected_rows() -> None:
    artifact, *_ = _observe()
    data = deepcopy(artifact.to_data())
    score = data["shards"][0]["slot_scores"][0]
    score["interval"] = {"lower": 4, "upper": 4}
    score["score_digest"] = canonical_digest(
        {key: value for key, value in score.items() if key != "score_digest"}
    )
    shard = data["shards"][0]
    shard["shard_digest"] = canonical_digest(
        {key: value for key, value in shard.items() if key != "shard_digest"}
    )
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    with pytest.raises(ObjectBongardRubricObserverError, match="model payload"):
        ObjectBongardRubricObserverArtifact.from_data(data)


def test_transport_failure_is_error_evidence_never_absence() -> None:
    artifact, *_ = _observe(fail=True)
    assert artifact.object_observations
    assert all(
        item.state is RubricObservationState.ERROR
        for item in artifact.object_observations
    )
    assert artifact.canonical_scene_observation is not None
    assert artifact.canonical_scene_observation.state is RubricObservationState.ERROR


def test_prompt_binds_exact_scale_and_has_no_experimental_role_words() -> None:
    _, hypotheses, _, spec = _inputs()
    prompt = object_bongard_rubric_observer_prompt(
        spec, hypotheses.atlas_sheets[0]
    )
    for _, meaning in RUBRIC_ORDINAL_LEVEL_ANCHORS:
        assert meaning in prompt
    lowered = prompt.lower()
    for word in ("positive", "negative", "query", "formula", "predicate"):
        assert word not in lowered


def test_observer_has_no_lean_import() -> None:
    source_path = Path(__file__).parents[1] / "object_bongard_rubric_observer.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any("lean" in item.lower() for item in imports)
