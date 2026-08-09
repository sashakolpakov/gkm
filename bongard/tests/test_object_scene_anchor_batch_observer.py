from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import re
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import pytest

from bongard import object_scene_anchor_batch_observer as batch_runtime
from bongard import transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_atlas import render_object_scene_anchor_atlas
from bongard.object_scene_anchor_batch_observer import (
    OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
    ObjectSceneAnchorBatchCapacityGap,
    ObjectSceneAnchorBatchObserverArtifact,
    ObjectSceneAnchorBatchObserverError,
    ObjectSceneAnchorBatchObserverInput,
    ObjectSceneAnchorBatchObserverPlan,
    _expected_records,
    _payload_cells,
    freeze_object_scene_anchor_batch_observer_plan,
    object_scene_anchor_batch_observer_output_schema,
    object_scene_anchor_batch_observer_prompt,
    observe_object_scene_anchor_batches_twice,
    verify_object_scene_anchor_batch_observer_artifact,
)
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingSpec,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_catalog import _make_entry
from bongard.object_scene_anchor_observer import (
    OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS,
    ObjectSceneAnchorObserverVocabulary,
    ObjectSceneAnchorObserverVocabularyEntry,
    _vocabulary_content,
    prepare_object_scene_anchor_observer_inputs,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    _manifest_content,
)
from bongard.object_scene_anchor_salience import extract_object_scene_anchor_salience
from bongard.object_scene_visual_frontend import OBJECT_SCENE_CANONICAL_SCENARIO_ID
from bongard.prototype_object_hypotheses import _crop_pixel_digest
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import CodexReceipt, CodexStructuredResult


LAUNCHER_DIGEST = "a" * 64
MODEL = transport_runtime.DEFAULT_CODEX_MODEL
EFFORT = "medium"


def _grayscale_png(luminance: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(luminance, mode="L").save(output, format="PNG", optimize=False)
    return output.getvalue()


def _vocabulary() -> ObjectSceneAnchorObserverVocabulary:
    rows = (
        ("shape_appearance", "A cross-like figure has four straight arms."),
        ("marking_pattern", "One arm is visibly darker than another arm."),
    )
    semantic = sorted(
        (
            canonical_digest(
                {
                    "schema": "gkm.object-scene-anchor-card-witness.v1",
                    "kind": kind,
                    "statement": statement,
                }
            ),
            kind,
            statement,
        )
        for kind, statement in rows
    )
    entries = tuple(
        ObjectSceneAnchorObserverVocabularyEntry.create(
            f"witness_{index:02d}", kind, statement, digest
        )
        for index, (digest, kind, statement) in enumerate(semantic)
    )
    provisional = object.__new__(ObjectSceneAnchorObserverVocabulary)
    object.__setattr__(provisional, "entries", entries)
    return ObjectSceneAnchorObserverVocabulary(
        entries, canonical_digest(_vocabulary_content(provisional))
    )


def _input(index: int, vocabulary: ObjectSceneAnchorObserverVocabulary, *, part: bool = False):
    # Each synthetic panel has one object, so its panel-local canonical ID is 0.
    object_id = "object_0000"
    mask = np.zeros((50, 50), dtype=np.bool_)
    mask[25, 8:43] = True
    mask[8:43, 25] = True
    strength = np.zeros(mask.shape, dtype=np.uint8)
    strength[mask] = 180
    strength[25, 25:43] = 255
    salience = extract_object_scene_anchor_salience(mask, object_id)
    crop_pixel_digest = _crop_pixel_digest(strength)
    digit = f"{(index % 15) + 1:x}"
    receipt = SimpleNamespace(
        object_id=object_id,
        receipt_digest=digit * 64,
        lineage_id="lineage-00000000",
        lineage_digest="2" * 64,
        scenario_id=OBJECT_SCENE_CANONICAL_SCENARIO_ID,
        hypothesis_id="hypothesis-00000000",
        hypothesis_digest="3" * 64,
        masked_crop_pixel_digest=crop_pixel_digest,
    )
    entry = _make_entry(
        inventory_index=0, receipt=receipt, mask=mask, salience=salience
    )
    panel_values = {
        "panel_digest": hashlib.sha256(f"panel-{index}".encode()).hexdigest(),
        "width_pixels": 80,
        "height_pixels": 80,
        "inventory_digest": hashlib.sha256(f"inventory-{index}".encode()).hexdigest(),
        "proposal_count": 1,
        "object_ids": (object_id,),
        "object_decisions": (entry.decision_manifest,),
    }
    provisional = object.__new__(ObjectSceneAnchorPanelDecisionManifest)
    for name, item in panel_values.items():
        object.__setattr__(provisional, name, item)
    panel = ObjectSceneAnchorPanelDecisionManifest(
        **panel_values,
        manifest_digest=canonical_digest(_manifest_content(provisional)),
    )
    atlas, atlas_png = render_object_scene_anchor_atlas(entry.decision_manifest)
    assert atlas_png is not None
    spec = ObjectSceneAnchorBindingSpec.part() if part else ObjectSceneAnchorBindingSpec.entity()
    catalog = build_object_scene_anchor_binding_catalog(
        entry.decision_manifest, spec, expected_object_id=object_id
    )
    crop_png = _grayscale_png(255 - strength)
    preparation = prepare_object_scene_anchor_observer_inputs(
        crop_png,
        catalog_entry=entry,
        panel_manifest=panel,
        atlas=atlas,
        atlas_png_bytes=atlas_png,
        catalog=catalog,
        vocabulary=vocabulary,
    )
    return ObjectSceneAnchorBatchObserverInput(preparation, crop_png, atlas_png)


def _payload(batch, vocabulary, state: str = "P") -> dict[str, object]:
    reason = {"P": "visible_match", "A": "visible_mismatch", "I": "unclear_geometry"}[state]
    return {
        "cells": [
            {
                "subject_id": subject.subject_alias,
                "catalog_id": catalog.catalog_alias,
                "binding_id": binding.binding_id,
                "witness_id": witness.witness_id,
                "state": state,
                "reason_code": reason,
            }
            for subject, catalog, binding, _locator, witness in _expected_records(batch, vocabulary)
        ]
    }


def _unique_receipt(receipt: CodexReceipt, index: int) -> CodexReceipt:
    data = receipt.to_dict()
    data["thread_id"] = f"00000000-0000-4000-8000-{index + 1:012d}"
    data["event_stream_digest"] = f"{index + 1:x}" * 64
    body = {key: value for key, value in data.items() if key != "receipt_digest"}
    data["receipt_digest"] = transport_runtime._digest(body)
    transport_runtime.validate_codex_receipt(data)
    return CodexReceipt(
        **{
            **data,
            "event_types": tuple(data["event_types"]),
            "item_types": tuple(data["item_types"]),
        }
    )


def _transport(plan, outcomes):
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fake(prompt, paths, names, schema, **kwargs):
        index = len(calls)
        calls.append((tuple(paths), tuple(names)))
        outcome = outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        batch = plan.batches[index // 2]
        payload = _payload(batch, plan.vocabulary, outcome)
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
            command_fixture=f"batch observer call {index}",
        )
        return CodexStructuredResult(payload, _unique_receipt(receipt, index))

    return fake, calls


@pytest.fixture(scope="module")
def campaign():
    vocabulary = _vocabulary()
    inputs = [_input(index, vocabulary) for index in range(24)]
    # A second binding specification shares view 0 and therefore adds no image.
    inputs.append(_input(0, vocabulary, part=True))
    plan = freeze_object_scene_anchor_batch_observer_plan(inputs)
    fake, calls = _transport(plan, ["P", "A", RuntimeError("offline"), "A"])
    model_catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    observation_plan = "sha256:" + canonical_digest({"batch_observer": 1})
    artifact = observe_object_scene_anchor_batches_twice(
        inputs,
        plan=plan,
        expected_plan_digest=plan.plan_digest,
        observation_plan_digest=observation_plan,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=attestation,
        transport=fake,
    )
    return inputs, plan, artifact, calls, observation_plan


def test_24_views_partition_into_two_batches_and_four_calls(campaign) -> None:
    _inputs, plan, artifact, calls, _observation_plan = campaign
    assert plan.view_count == 24
    assert plan.view_presentation_count == 24
    assert plan.catalog_count == 25
    assert plan.cell_count == sum(item.cell_count for item in plan.batches) == 56
    assert [item.view_count for item in plan.batches] == [16, 8]
    assert [item.image_count for item in plan.batches] == [32, 16]
    assert sorted(
        len(subject.catalogs) for batch in plan.batches for subject in batch.subjects
    ) == [1] * 23 + [2]
    assert sum(len(subject.catalogs) for batch in plan.batches for subject in batch.subjects) == 25
    assert len(calls) == artifact.physical_call_count == 4
    assert [len(names) for _paths, names in calls] == [32, 32, 16, 16]


def test_prompt_is_role_blind_and_rectangle_is_exact(campaign) -> None:
    _inputs, plan, _artifact, _calls, _observation_plan = campaign
    prompt = object_scene_anchor_batch_observer_prompt(plan.batches[0], plan.vocabulary)
    schema = object_scene_anchor_batch_observer_output_schema(
        plan.batches[0], plan.vocabulary
    )
    assert f"exactly {plan.batches[0].cell_count} cells" in prompt.lower()
    assert str(OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS) in prompt
    assert (
        schema["properties"]["cells"]["description"]
        == f"Exactly {plan.batches[0].cell_count} cells in the listed order; "
        f"the fixed batch limit is {OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS}."
    )
    for forbidden in (
        "target", "foil", "support", "contrast", "candidate", "formula", "query", "label"
    ):
        assert re.search(rf"\b{forbidden}s?\b", prompt, re.I) is None
    payload = _payload(plan.batches[0], plan.vocabulary)
    cells = _payload_cells(payload, plan.batches[0], plan.vocabulary)
    assert len(cells) == plan.batches[0].cell_count
    reordered = deepcopy(payload)
    reordered["cells"][0], reordered["cells"][1] = reordered["cells"][1], reordered["cells"][0]
    with pytest.raises(ObjectSceneAnchorBatchObserverError, match="order"):
        _payload_cells(reordered, plan.batches[0], plan.vocabulary)


def test_failures_and_disagreement_never_become_absence(campaign) -> None:
    _inputs, _plan, artifact, _calls, _observation_plan = campaign
    assert all(
        cell.disposition is Disposition.INDETERMINATE
        for cell in artifact.results[0].merged_cells
    )
    assert artifact.results[1].passes[0].status == "transport_error"
    assert all(
        cell.disposition is Disposition.ERROR
        for cell in artifact.results[1].merged_cells
    )
    assert not any(
        cell.disposition is Disposition.CERTIFIED_ABSENT
        for cell in artifact.results[1].passes[0].cells
    )


def test_strict_roundtrip_cold_replay_and_resealed_tamper_rejection(campaign) -> None:
    inputs, plan, artifact, _calls, observation_plan = campaign
    assert ObjectSceneAnchorBatchObserverArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_scene_anchor_batch_observer_artifact(
        artifact,
        inputs,
        expected_artifact_digest=artifact.artifact_digest,
        expected_plan_digest=plan.plan_digest,
        expected_observation_plan_digest=observation_plan,
    ) == artifact

    omitted = deepcopy(artifact.to_data())
    omitted["results"][0]["passes"][0]["cells"].pop()
    pass_data = omitted["results"][0]["passes"][0]
    pass_data["pass_digest"] = canonical_digest(
        {key: value for key, value in pass_data.items() if key != "pass_digest"}
    )
    result_data = omitted["results"][0]
    result_data["result_digest"] = canonical_digest(
        {key: value for key, value in result_data.items() if key != "result_digest"}
    )
    omitted["artifact_digest"] = canonical_digest(
        {key: value for key, value in omitted.items() if key != "artifact_digest"}
    )
    with pytest.raises(ObjectSceneAnchorBatchObserverError, match="structure|rectangle"):
        ObjectSceneAnchorBatchObserverArtifact.from_data(omitted)

    reordered = deepcopy(plan.to_data())
    reordered["batches"][0]["subjects"].reverse()
    batch = reordered["batches"][0]
    batch["batch_digest"] = canonical_digest(
        {key: value for key, value in batch.items() if key != "batch_digest"}
    )
    reordered["plan_digest"] = canonical_digest(
        {key: value for key, value in reordered.items() if key != "plan_digest"}
    )
    with pytest.raises(ObjectSceneAnchorBatchObserverError, match="subjects"):
        ObjectSceneAnchorBatchObserverPlan.from_data(reordered)


def test_cell_cap_covers_one_maximal_indivisible_preparation() -> None:
    assert OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS == 17 * 32 == 544
    assert OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS >= OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS


def test_exact_cell_boundary_is_order_independent(monkeypatch) -> None:
    monkeypatch.setattr(batch_runtime, "OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS", 10)
    vocabulary = _vocabulary()
    inputs = (
        _input(101, vocabulary),
        _input(101, vocabulary, part=True),
    )
    forward = freeze_object_scene_anchor_batch_observer_plan(inputs)
    reverse = freeze_object_scene_anchor_batch_observer_plan(tuple(reversed(inputs)))

    assert forward == reverse
    assert len(forward.batches) == 1
    assert forward.view_count == forward.view_presentation_count == 1
    assert forward.catalog_count == 2
    assert forward.cell_count == forward.batches[0].cell_count == 10
    assert len(forward.preparations) == 2


def test_same_view_splits_without_pruning_and_replays_exact_images(monkeypatch) -> None:
    monkeypatch.setattr(batch_runtime, "OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS", 9)
    vocabulary = _vocabulary()
    inputs = (
        _input(102, vocabulary),
        _input(102, vocabulary, part=True),
    )
    plan = freeze_object_scene_anchor_batch_observer_plan(inputs)

    assert plan == freeze_object_scene_anchor_batch_observer_plan(
        tuple(reversed(inputs))
    )
    assert len(plan.batches) == 2
    assert plan.view_count == 1
    assert plan.view_presentation_count == 2
    assert plan.to_data()["repeated_view_presentation_count"] == 1
    assert [item.view_count for item in plan.batches] == [1, 1]
    assert [item.image_count for item in plan.batches] == [2, 2]
    assert sorted(item.cell_count for item in plan.batches) == [2, 8]
    assert len(plan.preparations) == plan.catalog_count == 2
    assert {item.preparation_digest for item in plan.preparations} == {
        item.preparation.preparation_digest for item in inputs
    }
    assert (
        plan.batches[0].subjects[0].view_digest
        == plan.batches[1].subjects[0].view_digest
    )

    fake, calls = _transport(plan, ["P", "P", "A", "A"])
    model_catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    observation_plan = "sha256:" + canonical_digest({"split_same_view": 1})
    artifact = observe_object_scene_anchor_batches_twice(
        inputs,
        plan=plan,
        expected_plan_digest=plan.plan_digest,
        observation_plan_digest=observation_plan,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=attestation,
        transport=fake,
    )
    assert len(calls) == 4
    assert [names for _paths, names in calls] == [
        ("subject_00_object.png", "subject_00_anchors.png")
    ] * 4
    assert (
        artifact.results[0].passes[0].presentation
        == artifact.results[1].passes[0].presentation
    )
    assert verify_object_scene_anchor_batch_observer_artifact(
        artifact,
        inputs,
        expected_artifact_digest=artifact.artifact_digest,
        expected_plan_digest=plan.plan_digest,
        expected_observation_plan_digest=observation_plan,
    ) == artifact


def test_oversized_indivisible_preparation_is_a_typed_gap_before_calls(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batch_runtime, "OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS", 7)
    vocabulary = _vocabulary()
    oversized = _input(103, vocabulary, part=True)

    with pytest.raises(ObjectSceneAnchorBatchCapacityGap) as captured:
        freeze_object_scene_anchor_batch_observer_plan((oversized,))

    assert captured.value.preparation_digest == oversized.preparation.preparation_digest
    assert captured.value.cell_count == oversized.preparation.cell_count == 8
    assert captured.value.maximum_cell_count == 7
