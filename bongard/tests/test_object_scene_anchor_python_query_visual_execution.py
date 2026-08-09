"""Synthetic released-panel tests for the Python query visual executor."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from bongard.evidence import Disposition
from bongard.object_scene_anchor_batch_observer import (
    _expected_records,
    object_scene_anchor_batch_observer_prompt,
    observe_object_scene_anchor_batches_twice,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorSelectionCommitment,
    freeze_object_scene_anchor_python_predicate,
)
from bongard.object_scene_anchor_python_query_visual_execution import (
    ObjectSceneAnchorPythonQueryPanelInput,
    ObjectSceneAnchorPythonQueryVisualPlan,
    ObjectSceneAnchorPythonQueryVisualResult,
    build_object_scene_anchor_python_query_visual_plan,
    cold_verify_object_scene_anchor_python_query_visual_result,
    finalize_object_scene_anchor_python_query_visual_execution,
    verify_object_scene_anchor_python_query_visual_runtime,
)
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.tests.test_object_scene_anchor_exposed_query_gate import _seal_record
from bongard.tests.test_object_scene_anchor_python_predicate import _version_fixture
from bongard.tests.test_object_scene_anchor_support_observation_join import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    _panel_png,
    _unique_receipt,
)
from bongard.transport import CodexStructuredResult


def _address(character: str) -> str:
    return "sha256:" + character * 64


def _predicate():
    version, _language, _manifests = _version_fixture(
        lambda *_: Disposition.CERTIFIED_ABSENT
    )
    selected = next(
        item for item in version.candidates if len(item.witness_digests) == 2
    )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selected.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest="9" * 64,
    )
    return freeze_object_scene_anchor_python_predicate(version, selection)


def _contains_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_bytes(item) for item in value)
    return False


def _observe(runtime):
    plan = runtime.plan.batch_plan
    assert plan is not None
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        batch = plan.batches[calls // 2]
        payload = {
            "cells": [
                {
                    "subject_id": subject.subject_alias,
                    "catalog_id": catalog.catalog_alias,
                    "binding_id": binding.binding_id,
                    "witness_id": witness.witness_id,
                    "state": "P",
                    "reason_code": "visible_match",
                }
                for subject, catalog, binding, _locator, witness
                in _expected_records(batch, plan.vocabulary)
            ]
        }
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
            command_fixture=f"query visual call {calls}",
        )
        result = CodexStructuredResult(
            payload, _unique_receipt(receipt, calls)
        )
        calls += 1
        return result

    model_catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    artifact = observe_object_scene_anchor_batches_twice(
        runtime.batch_inputs,
        plan=plan,
        expected_plan_digest=plan.plan_digest,
        observation_plan_digest=runtime.plan.observation_context_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=attestation,
        transport=transport,
    )
    return artifact, calls


def test_released_query_observes_only_p_catalogs_twice_and_predicts() -> None:
    predicate = _predicate()
    official_panel_id = "bd/bd_query_visual_secret/0/6.png"
    released = _seal_record(official_panel_id, _panel_png(0))
    panel_input = ObjectSceneAnchorPythonQueryPanelInput(
        released, "panel_000", _address("8")
    )
    runtime = build_object_scene_anchor_python_query_visual_plan(
        panel_input, predicate
    )
    plan = runtime.plan

    assert plan.present_catalog_count > 0
    assert plan.batch_plan is not None
    assert plan.physical_call_count == 2 * len(plan.batch_plan.batches)
    assert not _contains_bytes(plan.to_data())
    assert ObjectSceneAnchorPythonQueryVisualPlan.from_data(plan.to_data()) == plan
    assert tuple(
        (item.kind, item.statement, item.witness_digest)
        for item in plan.local_observer_vocabulary.entries
    ) == tuple(
        (item.kind, item.statement, item.witness_digest)
        for item in plan.query_vocabulary.entries
    )
    assert all(
        official_panel_id not in object_scene_anchor_batch_observer_prompt(
            batch, plan.local_observer_vocabulary
        )
        for batch in plan.batch_plan.batches
    )
    assert official_panel_id not in str(plan.batch_plan.to_data())
    assert verify_object_scene_anchor_python_query_visual_runtime(
        runtime,
        panel_input=panel_input,
        predicate=predicate,
        expected_plan_digest=plan.plan_digest,
    ) == runtime

    artifact, calls = _observe(runtime)
    assert calls == plan.physical_call_count == 2
    assert official_panel_id not in str(artifact.to_data())
    result = finalize_object_scene_anchor_python_query_visual_execution(
        plan, artifact
    )
    assert result.query_evaluation.disposition is Disposition.PRESENT
    assert result.prediction.query_disposition is Disposition.PRESENT
    assert not _contains_bytes(result.to_data())
    assert ObjectSceneAnchorPythonQueryVisualResult.from_data(
        result.to_data()
    ) == result
    assert cold_verify_object_scene_anchor_python_query_visual_result(
        result,
        plan=plan,
        artifact=artifact,
        expected_result_digest=result.result_digest,
    ) == result


def test_zero_p_query_makes_zero_calls_and_certifies_absence() -> None:
    predicate = _predicate()
    image = Image.new("RGB", (64, 64), "white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    released = _seal_record(
        "bd/bd_query_visual_blank/1/6.png", output.getvalue()
    )
    panel_input = ObjectSceneAnchorPythonQueryPanelInput(
        released, "panel_001", _address("7")
    )
    runtime = build_object_scene_anchor_python_query_visual_plan(
        panel_input, predicate
    )

    assert runtime.plan.present_catalog_count == 0
    assert runtime.plan.batch_plan is None
    assert runtime.plan.batch_plan_digest is None
    assert runtime.plan.physical_call_count == 0
    assert runtime.batch_inputs == ()
    result = finalize_object_scene_anchor_python_query_visual_execution(
        runtime.plan, None
    )
    assert result.physical_call_count == 0
    assert result.batch_artifact_digest is None
    assert result.query_evaluation.disposition is Disposition.CERTIFIED_ABSENT
    assert result.prediction.query_disposition is Disposition.CERTIFIED_ABSENT
    assert cold_verify_object_scene_anchor_python_query_visual_result(
        result,
        plan=runtime.plan,
        artifact=None,
        expected_result_digest=result.result_digest,
    ) == result
