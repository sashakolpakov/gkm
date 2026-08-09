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

from bongard import transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_atlas import render_object_scene_anchor_atlas
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingSpec,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_catalog import _make_entry
from bongard.object_scene_anchor_observer import (
    ObjectSceneAnchorObserverArtifact,
    ObjectSceneAnchorObserverError,
    ObjectSceneAnchorObserverVocabulary,
    ObjectSceneAnchorObserverVocabularyEntry,
    object_scene_anchor_observer_output_schema,
    object_scene_anchor_observer_prompt,
    observe_object_scene_anchor_catalog_twice,
    prepare_object_scene_anchor_observer_inputs,
    verify_object_scene_anchor_observer_artifact,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    _manifest_content,
)
from bongard.object_scene_anchor_salience import (
    extract_object_scene_anchor_salience,
)
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_CANONICAL_SCENARIO_ID,
)
from bongard.prototype_object_hypotheses import _crop_pixel_digest
from bongard.transport import CodexReceipt, CodexStructuredResult
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)


LAUNCHER_DIGEST = "a" * 64
MODEL = transport_runtime.DEFAULT_CODEX_MODEL
EFFORT = "medium"


def _grayscale_png(luminance: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(luminance, mode="L").save(output, format="PNG", optimize=False)
    return output.getvalue()


def _scene():
    mask = np.zeros((50, 50), dtype=np.bool_)
    mask[25, 8:43] = True
    mask[8:43, 25] = True
    strength = np.zeros(mask.shape, dtype=np.uint8)
    strength[mask] = 180
    # Preserve a visible full-style intensity change that the Boolean anchor
    # graph intentionally does not contain.
    strength[25, 25:43] = 255
    salience = extract_object_scene_anchor_salience(mask, "object_0000")
    crop_pixel_digest = _crop_pixel_digest(strength)
    receipt = SimpleNamespace(
        object_id="object_0000",
        receipt_digest="1" * 64,
        lineage_id="lineage-00000000",
        lineage_digest="2" * 64,
        scenario_id=OBJECT_SCENE_CANONICAL_SCENARIO_ID,
        hypothesis_id="hypothesis-00000000",
        hypothesis_digest="3" * 64,
        masked_crop_pixel_digest=crop_pixel_digest,
    )
    entry = _make_entry(
        inventory_index=0,
        receipt=receipt,
        mask=mask,
        salience=salience,
    )
    panel_values = {
        "panel_digest": "4" * 64,
        "width_pixels": 80,
        "height_pixels": 80,
        "inventory_digest": "5" * 64,
        "proposal_count": 1,
        "object_ids": ("object_0000",),
        "object_decisions": (entry.decision_manifest,),
    }
    provisional = object.__new__(ObjectSceneAnchorPanelDecisionManifest)
    for name, item in panel_values.items():
        object.__setattr__(provisional, name, item)
    panel_manifest = ObjectSceneAnchorPanelDecisionManifest(
        **panel_values,
        manifest_digest=canonical_digest(_manifest_content(provisional)),
    )
    atlas, atlas_png = render_object_scene_anchor_atlas(entry.decision_manifest)
    assert atlas_png is not None
    catalog = build_object_scene_anchor_binding_catalog(
        entry.decision_manifest,
        ObjectSceneAnchorBindingSpec.entity(),
        expected_object_id="object_0000",
    )
    crop_png = _grayscale_png(255 - strength)
    vocabulary = _vocabulary(
        (
            ("shape_appearance", "A cross-like figure has four straight arms."),
            ("marking_pattern", "One arm is visibly darker than another arm."),
        )
    )
    return entry, panel_manifest, crop_png, atlas, atlas_png, catalog, vocabulary


def _vocabulary(
    rows: tuple[tuple[str, str], ...],
) -> ObjectSceneAnchorObserverVocabulary:
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
    from bongard.object_scene_anchor_observer import _vocabulary_content

    return ObjectSceneAnchorObserverVocabulary(
        entries,
        canonical_digest(_vocabulary_content(provisional)),
    )


def _prepared(scene):
    entry, panel, crop, atlas, atlas_png, catalog, vocabulary = scene
    return prepare_object_scene_anchor_observer_inputs(
        crop,
        catalog_entry=entry,
        panel_manifest=panel,
        atlas=atlas,
        atlas_png_bytes=atlas_png,
        catalog=catalog,
        vocabulary=vocabulary,
    )


def _payload(preparation, states: tuple[tuple[str, str], ...]) -> dict[str, object]:
    assert len(states) == preparation.cell_count
    return {
        "cells": [
            {
                "binding_id": locator.binding_id,
                "witness_id": witness.witness_id,
                "state": state,
                "reason_code": reason,
            }
            for (locator, witness), (state, reason) in zip(
                (
                    (locator, witness)
                    for locator in preparation.locators
                    for witness in preparation.vocabulary.entries
                ),
                states,
                strict=True,
            )
        ]
    }


def _unique_receipt(
    receipt: CodexReceipt, index: int
) -> CodexReceipt:
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


def _fake_transport(payloads: list[object]):
    calls: list[tuple[str, tuple[str, ...], tuple[str, ...], Mapping[str, Any]]] = []

    def fake(prompt, paths, names, schema, **kwargs):
        index = len(calls)
        calls.append((prompt, tuple(paths), tuple(names), schema))
        value = payloads[index]
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, Mapping)
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            value,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
            command_fixture=f"anchor observer pass {index}",
        )
        return CodexStructuredResult(dict(value), _unique_receipt(receipt, index))

    return fake, calls


def _observe(scene, payloads):
    entry, panel, crop, atlas, atlas_png, catalog, vocabulary = scene
    preparation = _prepared(scene)
    model_catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    fake, calls = _fake_transport(payloads)
    plan = "sha256:" + canonical_digest({"neutral_observer_plan": 1})
    artifact = observe_object_scene_anchor_catalog_twice(
        crop,
        catalog_entry=entry,
        panel_manifest=panel,
        atlas=atlas,
        atlas_png_bytes=atlas_png,
        catalog=catalog,
        vocabulary=vocabulary,
        expected_panel_manifest_digest=panel.manifest_digest,
        expected_crop_png_digest=preparation.crop_png_digest,
        expected_crop_pixel_digest=entry.masked_crop_pixel_digest,
        expected_atlas_artifact_digest=atlas.artifact_digest,
        expected_atlas_png_digest=hashlib.sha256(atlas_png).hexdigest(),
        expected_catalog_digest=catalog.catalog_digest,
        expected_vocabulary_digest=vocabulary.vocabulary_digest,
        observation_plan_digest=plan,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=attestation,
        transport=fake,
    )
    return artifact, calls, plan


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_preparation_binds_full_style_crop_but_drops_full_entry_identity() -> None:
    scene = _scene()
    entry, panel, crop, atlas, atlas_png, catalog, vocabulary = scene
    preparation = _prepared(scene)

    assert preparation.crop_pixel_digest == entry.masked_crop_pixel_digest
    assert preparation.crop_width_pixels == entry.crop_width_pixels
    assert preparation.crop_height_pixels == entry.crop_height_pixels
    assert preparation.panel_manifest == panel
    assert preparation.catalog == catalog
    assert preparation.cell_count == len(catalog.bindings) * len(vocabulary.entries)
    assert "entry_digest" not in set(_all_keys(preparation.to_data()))
    assert "salience_artifact_digest" not in set(_all_keys(preparation.to_data()))

    changed = bytearray(crop)
    changed[-20] ^= 1
    with pytest.raises(Exception):
        prepare_object_scene_anchor_observer_inputs(
            bytes(changed),
            catalog_entry=entry,
            panel_manifest=panel,
            atlas=atlas,
            atlas_png_bytes=atlas_png,
            catalog=catalog,
            vocabulary=vocabulary,
        )


def test_prompt_and_schema_are_role_blind_and_exactly_rectangular() -> None:
    preparation = _prepared(_scene())
    prompt = object_scene_anchor_observer_prompt(
        preparation.locators, preparation.vocabulary
    )
    schema = object_scene_anchor_observer_output_schema(
        preparation.locators, preparation.vocabulary
    )
    envelope = prompt + "\n" + str(schema)

    for forbidden in ("target", "foil", "candidate", "formula", "negation"):
        assert re.search(rf"\b{forbidden}s?\b", envelope, re.I) is None
    assert "side0" not in envelope and "side1" not in envelope
    assert [item.witness_id for item in preparation.vocabulary.entries] == [
        "witness_00",
        "witness_01",
    ]


def test_two_independent_passes_merge_same_exact_cells_before_logic() -> None:
    scene = _scene()
    preparation = _prepared(scene)
    first = _payload(
        preparation,
        (("P", "visible_match"), ("A", "visible_mismatch")),
    )
    second = _payload(
        preparation,
        (("P", "visible_match"), ("P", "visible_match")),
    )
    artifact, calls, _plan = _observe(scene, [first, second])

    assert len(calls) == 2
    assert artifact.physical_call_count == 2
    assert artifact.passes[0].receipt is not None
    assert artifact.passes[1].receipt is not None
    assert artifact.passes[0].receipt.thread_id != artifact.passes[1].receipt.thread_id
    assert [item.disposition for item in artifact.merged_cells] == [
        Disposition.PRESENT,
        Disposition.INDETERMINATE,
    ]
    assert [item.reason_code for item in artifact.merged_cells] == [
        "two_pass_visible_match",
        "two_pass_disagreement",
    ]
    assert all(
        cell.locator.binding_digest == preparation.catalog.bindings[0].binding_digest
        for cell in artifact.merged_cells
    )


def test_transport_or_uncertain_vision_never_becomes_absence() -> None:
    scene = _scene()
    preparation = _prepared(scene)
    uncertain = _payload(
        preparation,
        (("I", "unclear_geometry"), ("I", "image_quality")),
    )
    artifact, calls, _ = _observe(
        scene, [RuntimeError("offline failure"), uncertain]
    )

    assert len(calls) == 2
    assert artifact.passes[0].status == "transport_error"
    assert artifact.passes[1].status == "success"
    assert all(
        item.disposition is Disposition.ERROR for item in artifact.merged_cells
    )
    assert all(
        item.reason_code == "one_or_both_pass_error"
        for item in artifact.merged_cells
    )
    assert not any(
        item.disposition is Disposition.CERTIFIED_ABSENT
        for item in artifact.passes[0].cells
    )


def test_strict_roundtrip_cold_replay_and_omission_rejection() -> None:
    scene = _scene()
    preparation = _prepared(scene)
    payload = _payload(
        preparation,
        (("P", "visible_match"), ("I", "unclear_marking")),
    )
    artifact, _calls, plan = _observe(scene, [payload, payload])
    entry, panel, crop, atlas, atlas_png, catalog, vocabulary = scene

    assert ObjectSceneAnchorObserverArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_scene_anchor_observer_artifact(
        artifact,
        crop,
        catalog_entry=entry,
        panel_manifest=panel,
        atlas=atlas,
        atlas_png_bytes=atlas_png,
        catalog=catalog,
        vocabulary=vocabulary,
        expected_artifact_digest=artifact.artifact_digest,
        expected_observation_plan_digest=plan,
    ) == artifact

    omitted = deepcopy(artifact.to_data())
    omitted["passes"][0]["cells"].pop()
    # Even if an attacker reseals the immediate pass and outer records, the
    # exhaustive rectangle is recomputed from the embedded catalog/vocabulary.
    pass_body = {
        key: value
        for key, value in omitted["passes"][0].items()
        if key != "pass_digest"
    }
    omitted["passes"][0]["pass_digest"] = canonical_digest(pass_body)
    artifact_body = {
        key: value for key, value in omitted.items() if key != "artifact_digest"
    }
    omitted["artifact_digest"] = canonical_digest(artifact_body)
    with pytest.raises(ObjectSceneAnchorObserverError, match="rectangle"):
        ObjectSceneAnchorObserverArtifact.from_data(omitted)


def test_parser_failure_is_receipted_error_and_second_pass_still_runs() -> None:
    scene = _scene()
    preparation = _prepared(scene)
    malformed = {"cells": []}
    valid = _payload(
        preparation,
        (("A", "visible_mismatch"), ("A", "visible_mismatch")),
    )
    artifact, calls, _ = _observe(scene, [malformed, valid])

    assert len(calls) == 2
    assert artifact.passes[0].status == "parser_error"
    assert artifact.passes[0].receipt is not None
    assert artifact.passes[1].status == "success"
    assert all(
        item.disposition is Disposition.ERROR for item in artifact.merged_cells
    )
