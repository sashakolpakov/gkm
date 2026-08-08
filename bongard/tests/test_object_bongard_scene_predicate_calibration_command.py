"""Offline end-to-end tests for neutral scene-predicate calibration custody."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from bongard.canonical import canonical_digest
import bongard.object_bongard_scene_predicate_calibration_command as command
import bongard.object_scene_visual_frontend as frontend
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_COUNT_OBSERVABLE_IDS,
    OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS,
    ObjectSceneProposalInventory,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_scene_visual_frontend import _scene
from bongard.tests.test_prototype_scene_observer import _receipt
from bongard.transport import (
    CODEX_APPLY_PATCH_TOOL_TYPE,
    CODEX_EFFECTIVE_TOOL_MODE,
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    CODEX_TOOL_SURFACE_DIGEST,
    CODEX_TRANSPORT_POLICY_DIGEST,
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)
import bongard.transport as transport_module


LAUNCHER_DIGEST = "b" * 64
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(
    LAUNCHER_DIGEST
)


def _neutral_panel(index: int, exact_png_bytes: bytes) -> command._NeutralPanel:
    values: dict[str, Any] = {
        "ordinal": index,
        "blind_panel_id": f"calibration_panel_{index:02d}",
        "journal_task_id": f"bd_scene_calibration_{index:02d}",
        "task_id": f"historical_lineage_task_{index:02d}",
        "panel_id": f"historical_lineage_panel_{index:02d}",
        "released_record_digest": "sha256:" + f"{index + 1:064x}",
        "png_sha256": hashlib.sha256(exact_png_bytes).hexdigest(),
        "exact_png_bytes": exact_png_bytes,
    }
    provisional = object.__new__(command._NeutralPanel)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return command._NeutralPanel(
        **values,
        neutral_panel_digest=canonical_digest(
            command._neutral_panel_content(provisional)
        ),
    )


def test_registered_visual_blindness_allows_visual_orientation_prose_only() -> None:
    legitimate = SimpleNamespace(
        prompt=(
            "Compare the visible orientation of the oblique strokes; each lobe "
            "plays a different visual role in the bidirectional silhouette."
        ),
        output_schema={"type": "object", "properties": {}},
        presentation=(("panel.png", b"png"),),
    )
    command._assert_registered_visual_observer_blind(legitimate, "legitimate")

    for leaked in (
        "orientation_constraint=group0_positive",
        "historical role: 0",
        "support_role=1",
        "evaluate side 1 positive",
        "this belongs to group 0",
    ):
        prepared = SimpleNamespace(
            prompt=leaked,
            output_schema={"type": "object", "properties": {}},
            presentation=(("panel.png", b"png"),),
        )
        with pytest.raises(
            command.ObjectBongardScenePredicateCalibrationCommandError,
            match="leaks role or orientation-constraint metadata",
        ):
            command._assert_registered_visual_observer_blind(prepared, "leaked")


@pytest.fixture(scope="module")
def calibration_inputs() -> command._CalibrationInputs:
    # Build the expensive geometry once, then give that fixed synthetic
    # proposal catalog twelve distinct pixel commitments.  Frontend extraction
    # fidelity itself is covered by its dedicated tests; this fixture exercises
    # command custody and the real scene-IR boundary without twelve slow
    # extractor reruns.
    raws = tuple(_scene(index) for index in range(command.PANEL_COUNT))
    base_inventory, rendered = frontend._build_object_scene_inventory(raws[0])
    inventories_list: list[ObjectSceneProposalInventory] = []
    for raw in raws:
        panel_digest = hashlib.sha256(raw).hexdigest()
        lineage = replace(base_inventory.lineage_packet, panel_digest=panel_digest)
        values = {
            item.name: getattr(base_inventory, item.name)
            for item in fields(base_inventory)
            if item.name != "inventory_digest"
        }
        values.update(
            panel_digest=panel_digest,
            lineage_packet=lineage,
            lineage_packet_digest=lineage.digest(),
        )
        provisional = object.__new__(ObjectSceneProposalInventory)
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        inventories_list.append(
            ObjectSceneProposalInventory(
                **values,
                inventory_digest=canonical_digest(
                    frontend._inventory_content(provisional)
                ),
            )
        )
    panels = tuple(
        _neutral_panel(index, raw) for index, raw in enumerate(raws)
    )
    inventories = tuple(inventories_list)
    atlas = {
        panel.neutral_panel_digest: rendered for panel in panels
    }
    rows = tuple(
        {
            "ordinal": index,
            "blind_panel_id": panel.blind_panel_id,
            "neutral_panel_digest": panel.neutral_panel_digest,
            "historical_role": 0 if index < 6 else 1,
        }
        for index, panel in enumerate(panels)
    )
    source = SimpleNamespace(
        source_digest="d" * 64,
        historical_plan_file_sha256="e" * 64,
        historical_plan_record_digest="sha256:" + "f" * 64,
    )
    return command._CalibrationInputs(
        source=source,
        panels=panels,
        inventories=inventories,
        atlas_png_by_panel_digest=atlas,
        role_reveal_rows=rows,
        role_commitment_digest=command._role_commitment(rows),
    )


def _source_identities() -> list[dict[str, str]]:
    return [
        {"role": "offline_test_source", "sha256": "1" * 64},
        {"role": "scene_predicate_ir_source", "sha256": "2" * 64},
    ]


def _visual_payload(
    inventory: object,
    *,
    registered: bool,
    role: int,
    marker: str,
    panel_registered_witness_ids: Mapping[str, tuple[str, ...]] | None = None,
    entity_registered_witness_ids: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    panel_cards = (
        ()
        if panel_registered_witness_ids is None
        else tuple(panel_registered_witness_ids.items())
    )
    entity_cards = (
        ()
        if entity_registered_witness_ids is None
        else tuple(entity_registered_witness_ids.items())
    )
    rows: list[dict[str, object]] = []
    for crop in inventory.objects:
        rows.append(
            {
                "object_id": crop.object_id,
                "summary": f"outlined form with inspectable visible geometry {marker}",
                "counts": [
                    {
                        "observable_id": observable_id,
                        "state": "measured",
                        "lower_count": 1,
                        "upper_count": 1,
                        "evidence": "visible marks and boundaries were counted",
                    }
                    for observable_id in OBJECT_SCENE_COUNT_OBSERVABLE_IDS
                ],
                "observables": [
                    {
                        "observable_id": observable_id,
                        "state": (
                            "present"
                            if role == 0 and observable_id == "bird_like"
                            else "absent"
                        ),
                        "evidence": "the named appearance is directly inspectable",
                    }
                    for observable_id in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS
                ],
                "open_tags": (
                    []
                    if registered
                    else [
                        {
                            "tag": "bird-like object",
                            "state": "present",
                            "evidence": "the visible silhouette supports this phrase",
                        }
                    ]
                ),
                "registered_tags": (
                    [
                        {
                            "tag_id": tag_id,
                            "witness_cells": [
                                {
                                    "witness_id": witness_id,
                                    "state": (
                                        "present" if role == 0 else "absent"
                                    ),
                                    "evidence": (
                                        "the frozen witness was checked directly"
                                    ),
                                }
                                for witness_id in witness_ids
                            ],
                        }
                        for tag_id, witness_ids in entity_cards
                    ]
                    if registered
                    else []
                ),
            }
        )
    panel_phrase = (
        "bird-like object"
        if role == 0
        else "three-sided frame with distinct edge markers"
    )
    return {
        "panel": {
            "summary": f"complete visible composition {marker}",
            "open_tags": (
                []
                if registered
                else [
                    {
                        "tag": panel_phrase,
                        "state": "present",
                        "evidence": "the complete composition visibly supports the phrase",
                    }
                ]
            ),
            "registered_tags": (
                [
                    {
                        "tag_id": tag_id,
                        "witness_cells": [
                            {
                                "witness_id": witness_id,
                                "state": (
                                    "present" if role == 0 else "absent"
                                ),
                                "evidence": (
                                    "the frozen whole-panel witness was checked directly"
                                ),
                            }
                            for witness_id in witness_ids
                        ],
                    }
                    for tag_id, witness_ids in panel_cards
                ]
                if registered
                else []
            ),
        },
        "objects": rows,
    }


def _text_receipt(
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> CodexReceipt:
    schema_digest = canonical_digest(dict(schema))
    capture = next(
        row
        for row in NO_TOOLS_ATTESTATION.to_dict()["captures"]
        if row["modality"] == "text"
    )
    binding = {
        "model_catalog_digest": MODEL_CATALOG.raw_digest,
        "transport_policy_digest": CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": capture["normalized_command_digest"],
        "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        "tool_surface_attestation_digest": NO_TOOLS_ATTESTATION.attestation_digest,
    }
    causal = transport_module._causal_text_input_metadata(
        prompt, schema_digest, binding
    )
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": command.MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": command.REASONING_EFFORT,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": "00000000-0000-4000-8000-000000000091",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": "absent",
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "c" * 64,
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _bundle(
    registry: object, semantic_registry_proposal: object, *, accepted: bool
) -> dict[str, object]:
    formula_digest = "3" * 64
    atom_digest = "4" * 64
    evidence_digest = "5" * 64
    candidate_body: dict[str, object] = {
        "schema": "gkm.synthetic-scene-candidate.v1",
        "orientation": "group0_positive",
        "complexity": 1,
        "formula": {
            "node": "quantified",
            "quantifier": "exists",
            "atom": {"kind": "qualitative", "observable_id": "bird_like"},
            "atom_digest": atom_digest,
            "formula_digest": formula_digest,
        },
        "evidence_digest": evidence_digest,
    }
    candidate_digest = canonical_digest(candidate_body)
    candidate = {**candidate_body, "candidate_digest": candidate_digest}
    body: dict[str, object] = {
        "schema": command.IR_BUNDLE_SCHEMA,
        "ir_source_digest": "6" * 64,
        "algorithm_digest": "7" * 64,
        "registry_digest": registry.registry_digest,
        "registry_derivation_mode": "role_aware_semantic_concept_proposal",
        "registry_derivation_digest": semantic_registry_proposal.proposal_digest,
        "coverage_gate": {"passed": True, "covered_panel_count": 12},
        "selectivity_gate": {"passed": True, "separated_panel_count": 12},
        "repeatability_gate": {
            "passed": accepted,
            "repeat_tested_panel_count": 12 if accepted else 0,
        },
        "version_space": {
            "group0_positive": [candidate_digest] if accepted else [],
            "group1_positive": [],
            "complete": True,
        },
        "candidates": [candidate],
        "complete_survivor_digests": [candidate_digest] if accepted else [],
        "ranker_slate": (
            [
                {
                    "candidate_digest": candidate_digest,
                    "orientation": "group0_positive",
                    "complexity": 1,
                    "formula": {
                        "node": "quantified",
                        "quantifier": "exists",
                        "atom": {
                            "kind": "qualitative",
                            "observable_id": "bird_like",
                        },
                    },
                    "merged_support_summary": {
                        "present": 6,
                        "certified_absent": 6,
                        "indeterminate": 0,
                        "error": 0,
                    },
                }
            ]
            if accepted
            else []
        ),
        "omitted_survivors": [],
    }
    return {**body, "bundle_digest": canonical_digest(body)}


def _redigest_bundle(value: Mapping[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "bundle_digest"}
    return {**body, "bundle_digest": canonical_digest(body)}


def _install_offline_boundary(
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
    *,
    accepted: bool | None,
) -> None:
    monkeypatch.setattr(command, "_load_inputs", lambda _source_root: calibration_inputs)
    monkeypatch.setattr(command, "_source_identities", _source_identities)

    inventory_by_digest = {
        item.inventory_digest: item for item in calibration_inputs.inventories
    }
    atlas_by_inventory_digest = {
        inventory.inventory_digest: calibration_inputs.atlas_png_by_panel_digest[
            calibration_inputs.panels[index].neutral_panel_digest
        ]
        for index, inventory in enumerate(calibration_inputs.inventories)
    }

    def fast_verify_inventory(
        inventory,
        png_bytes,
        *,
        expected_inventory_digest=None,
        expected_atlas_png_by_name=None,
    ):
        assert hashlib.sha256(png_bytes).hexdigest() == inventory.panel_digest
        assert inventory_by_digest[inventory.inventory_digest] == inventory
        if expected_inventory_digest is not None:
            assert inventory.inventory_digest == expected_inventory_digest
        if expected_atlas_png_by_name is not None:
            assert dict(expected_atlas_png_by_name) == dict(
                atlas_by_inventory_digest[inventory.inventory_digest]
            )
        return inventory

    monkeypatch.setattr(
        frontend, "verify_object_scene_proposal_inventory", fast_verify_inventory
    )
    monkeypatch.setattr(
        frontend,
        "render_object_scene_proposal_atlas",
        lambda inventory, _png_bytes: atlas_by_inventory_digest[
            inventory.inventory_digest
        ],
    )

    if accepted is not None:
        def derive(**kwargs: object) -> dict[str, object]:
            return command._validate_ir_bundle(
                _bundle(
                    kwargs["registry"],
                    kwargs["semantic_registry_proposal"],
                    accepted=accepted,
                )
            )

        monkeypatch.setattr(command, "_derive_ir_bundle", derive)


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
    *,
    accepted: bool | None,
    semantic_valid: bool = True,
    semantic_extra_invalid: bool = False,
) -> tuple[
    command.VerifiedObjectBongardScenePredicateCalibration,
    list[str],
    list[str],
    list[str],
]:
    _install_offline_boundary(
        monkeypatch, calibration_inputs, accepted=accepted
    )
    by_panel_digest = {
        panel.png_sha256: (index, inventory)
        for index, (panel, inventory) in enumerate(
            zip(calibration_inputs.panels, calibration_inputs.inventories, strict=True)
        )
    }
    visual_calls: list[str] = []
    proposer_calls: list[str] = []
    ranker_calls: list[str] = []
    registered_prompts: dict[str, list[str]] = {
        "registered_a": [],
        "registered_b": [],
    }
    lock = Lock()

    def semantic_payload(prompt, schema):
        properties = schema["properties"]
        model_view = json.loads(
            prompt.split("Frozen descriptions:\n", maxsplit=1)[1]
        )

        def support_bindings(
            orientation: str, scope: str
        ) -> list[dict[str, str]]:
            rows_key = orientation.replace(
                "_positive", "_support_descriptions"
            )
            rows = {
                row["panel_alias"]: row
                for row in model_view[rows_key]
            }
            panel_aliases = model_view[
                "required_positive_binding_panels"
            ][orientation]
            schema_aliases = properties[orientation]["items"][
                "properties"
            ]["support_bindings"]["items"]["properties"][
                "panel_alias"
            ]["enum"]
            assert panel_aliases == schema_aliases
            return [
                {
                    "panel_alias": panel_alias,
                    "target_alias": (
                        "whole_panel"
                        if scope == "panel"
                        else rows[panel_alias]["proposal_atlas_map"][0][
                            "entity_alias"
                        ]
                    ),
                }
                for panel_alias in panel_aliases
            ]

        side0_bindings = support_bindings("side0_positive", "panel")
        side1_bindings = support_bindings("side1_positive", "panel")
        payload = {
            "side0_positive": [
                {
                    "scope": "panel",
                    "phrase": "bird-like object",
                    "required_witnesses": [
                        {
                            "kind": "shape_appearance",
                            "statement": (
                                "a compact body has two wing-like extensions"
                            ),
                        }
                    ],
                    "accepted_variants": [
                        "rounded wing tips count as equivalent extensions"
                    ],
                    "near_miss_boundaries": [
                        "a plain circular blob does not qualify"
                    ],
                    "support_bindings": side0_bindings,
                }
            ],
            "side1_positive": [
                {
                    "scope": "panel",
                    "phrase": (
                        "three-sided frame with distinct edge markers"
                        if semantic_valid
                        else "not a valid affirmative concept"
                    ),
                    "required_witnesses": [
                        {
                            "kind": "shape_appearance",
                            "statement": (
                                "the outer form has three long boundary segments"
                            ),
                        },
                        {
                            "kind": "marking_pattern",
                            "statement": (
                                "distinct small markers appear along the outer boundary"
                            ),
                        },
                    ],
                    "accepted_variants": [],
                    "near_miss_boundaries": [
                        "an open two-segment angle does not qualify"
                    ],
                    "support_bindings": side1_bindings,
                }
            ],
        }
        if semantic_extra_invalid:
            payload["side0_positive"].append(
                {
                    "scope": "entity",
                    "phrase": "pointed or curved",
                    "required_witnesses": [
                        {
                            "kind": "shape_appearance",
                            "statement": "the visible outline has one pointed end",
                        }
                    ],
                    "accepted_variants": [],
                    "near_miss_boundaries": [],
                    "support_bindings": support_bindings(
                        "side0_positive", "entity"
                    ),
                }
            )
        return payload

    def named_transport(prompt, paths, names, schema, **_kwargs):
        assert (root / command.AUTHORIZATION_FILENAME).is_file()
        assert (root / command.PRECOMMIT_FILENAME).is_file()
        properties = schema["properties"]
        if {"side0_positive", "side1_positive"}.issubset(properties):
            assert (root / command.DISCOVERY_FREEZE_FILENAME).is_file()
            assert (root / command.ROLE_REVEAL_FILENAME).is_file()
            assert not (root / command.REGISTRY_FREEZE_FILENAME).exists()
            assert tuple(names) == tuple(
                name
                for index in range(command.PANEL_COUNT)
                for name in (
                    f"panel_{index:03d}.png",
                    f"panel_{index:03d}_objects_000.png",
                )
            )
            proposer_calls.append(prompt)
            payload = semantic_payload(prompt, schema)
            return CodexStructuredResult(
                payload, _receipt(prompt, paths, names, schema, payload)
            )
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        index, inventory = by_panel_digest[panel_digest]
        registered = "All open_tags arrays must be empty" in prompt
        if not registered:
            stage = "discovery"
            assert not (root / command.ROLE_REVEAL_FILENAME).exists()
            assert not (root / command.REGISTRY_FREEZE_FILENAME).exists()
        elif (root / command.EVALUATION_A_BATCH_FILENAME).exists():
            stage = "registered_b"
        else:
            stage = "registered_a"
            assert (root / command.ROLE_REVEAL_FILENAME).is_file()
            assert (root / command.SEMANTIC_PROPOSAL_RESULT_FILENAME).is_file()
            assert (root / command.REGISTRY_FREEZE_FILENAME).is_file()
        envelope = prompt + json.dumps(schema, sort_keys=True) + " ".join(names)
        for panel in calibration_inputs.panels:
            assert panel.task_id not in envelope
            assert panel.panel_id not in envelope
        if registered:
            for forbidden in (
                "group0_positive",
                "group1_positive",
                "side0_positive",
                "side1_positive",
                "bidirectional",
                "orientation_constraint",
                "historical_role",
                "support_role",
            ):
                assert forbidden not in envelope
        with lock:
            call_ordinal = visual_calls.count(stage)
            visual_calls.append(stage)
            if registered:
                registered_prompts[stage].append(prompt)
        panel_tag_schema = schema["properties"]["panel"]["properties"][
            "registered_tags"
        ]["items"]
        entity_tag_schema = schema["properties"]["objects"]["items"][
            "properties"
        ]["registered_tags"]["items"]
        panel_ids = tuple(
            panel_tag_schema["properties"]["tag_id"].get("enum", ())
        )
        entity_ids = tuple(
            entity_tag_schema["properties"]["tag_id"].get("enum", ())
        )
        panel_cards: dict[str, tuple[str, ...]] = {}
        entity_cards: dict[str, tuple[str, ...]] = {}
        if registered:
            assert set(panel_tag_schema["properties"]) == {
                "tag_id",
                "witness_cells",
            }
            assert set(entity_tag_schema["properties"]) == {
                "tag_id",
                "witness_cells",
            }
            frozen = json.loads(
                (root / command.REGISTRY_FREEZE_FILENAME).read_text("utf-8")
            )["registry"]
            tags_by_id = {item["tag_id"]: item for item in frozen["tags"]}
            panel_cards = {
                tag_id: tuple(
                    item["witness_id"]
                    for item in tags_by_id[tag_id]["required_witnesses"]
                )
                for tag_id in panel_ids
            }
            entity_cards = {
                tag_id: tuple(
                    item["witness_id"]
                    for item in tags_by_id[tag_id]["required_witnesses"]
                )
                for tag_id in entity_ids
            }
        payload = _visual_payload(
            inventory,
            registered=registered,
            role=0 if index < 6 else 1,
            marker=f"{stage}-{call_ordinal:02d}",
            panel_registered_witness_ids=panel_cards,
            entity_registered_witness_ids=entity_cards,
        )
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    def text_transport(prompt, schema, **_kwargs):
        properties = schema["properties"]
        assert set(properties) == {"selected_survivor_digest"}
        assert (root / command.RANK_INPUT_FREEZE_FILENAME).is_file()
        assert (root / command.ROLE_REVEAL_FILENAME).is_file()
        enum = properties["selected_survivor_digest"]["enum"]
        payload = {"selected_survivor_digest": enum[0]}
        ranker_calls.append(prompt)
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))

    verified = command.run_object_bongard_scene_predicate_calibration(
        root,
        source_root="offline-source",
        parallel_workers=4,
        minutes=3,
        expected_launcher_sha256=LAUNCHER_DIGEST,
        named_image_transport=named_transport,
        text_transport=text_transport,
        cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        catalog_snapshotter=lambda: MODEL_CATALOG,
        launcher_fingerprinter=lambda _executable, **_kwargs: {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        },
        runtime_attester=lambda **_kwargs: NO_TOOLS_ATTESTATION,
    )
    assert len(visual_calls) == command.VISUAL_CALL_COUNT == 36
    assert visual_calls.count("discovery") == 12
    assert visual_calls.count("registered_a") == 12
    assert visual_calls.count("registered_b") == 12
    assert len(registered_prompts["registered_a"]) == 12
    assert len(registered_prompts["registered_b"]) == 12
    assert set(registered_prompts["registered_a"]) == set(
        registered_prompts["registered_b"]
    )
    for prompt in registered_prompts["registered_a"]:
        if semantic_valid:
            assert "bird-like object" in prompt
            assert "three-sided frame with distinct edge markers" in prompt
            assert "witness_00 [shape_appearance]" in prompt
            assert "witness_00 [marking_pattern]" in prompt
            assert "witness_01 [shape_appearance]" in prompt
            assert "a compact body has two wing-like extensions" in prompt
            assert "distinct small markers appear along the outer boundary" in prompt
        assert "historical_role" not in prompt
        assert "side0_positive" not in prompt
        assert "side1_positive" not in prompt
        assert "support_bindings" not in prompt
        assert "target_alias" not in prompt
    return verified, visual_calls, proposer_calls, ranker_calls


def test_accepted_run_makes_exactly_38_calls_then_zero_call_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "accepted_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root, monkeypatch, calibration_inputs, accepted=True
    )

    assert verified.status == "accepted"
    assert verified.visual_fresh_call_count == 36
    assert verified.semantic_proposer_fresh_call_count == 1
    assert verified.ranker_fresh_call_count == 1
    assert verified.selected_survivor_digest is not None
    result = json.loads((root / command.RESULT_FILENAME).read_text("utf-8"))
    assert result["physical_model_call_count"] == 38
    assert result["typed_gap_status"] is None
    assert result["registry_derivation_mode"] == (
        "role_aware_semantic_concept_proposal"
    )
    assert result["roles_hidden_through_blind_discovery_freeze"] is True
    assert (
        result["roles_revealed_only_to_multimodal_semantic_proposer"] is True
    )
    assert "roles_revealed_only_to_zero_image_semantic_proposer" not in result
    assert result["benchmark_acceptance_authorized_registry"] is True
    assert result["exact_frequency_fallback_acceptance_authorized"] is False
    expected_schemas = {
        command.AUTHORIZATION_FILENAME: command.AUTHORIZATION_SCHEMA,
        command.PRECOMMIT_FILENAME: command.PRECOMMIT_SCHEMA,
        command.DISCOVERY_BATCH_FILENAME: command.DISCOVERY_BATCH_SCHEMA,
        command.DISCOVERY_FREEZE_FILENAME: command.DISCOVERY_FREEZE_SCHEMA,
        command.ROLE_REVEAL_FILENAME: command.ROLE_REVEAL_SCHEMA,
        command.SEMANTIC_PROPOSAL_INPUT_FILENAME: (
            command.SEMANTIC_PROPOSAL_INPUT_SCHEMA
        ),
        command.SEMANTIC_PROPOSAL_RESULT_FILENAME: (
            command.SEMANTIC_PROPOSAL_RESULT_SCHEMA
        ),
        command.REGISTRY_FREEZE_FILENAME: command.REGISTRY_FREEZE_SCHEMA,
        command.EVALUATION_A_BATCH_FILENAME: command.EVALUATION_BATCH_SCHEMA,
        command.EVALUATION_B_BATCH_FILENAME: command.EVALUATION_BATCH_SCHEMA,
        command.EVALUATION_FREEZE_FILENAME: command.EVALUATION_FREEZE_SCHEMA,
        command.ASSESSMENT_FILENAME: command.ASSESSMENT_SCHEMA,
        command.RANK_INPUT_FREEZE_FILENAME: command.RANK_INPUT_FREEZE_SCHEMA,
        command.RANK_RESULT_FILENAME: command.RANK_RESULT_SCHEMA,
        command.FORMULA_FREEZE_FILENAME: command.FORMULA_FREEZE_SCHEMA,
        command.REPLAY_FILENAME: command.REPLAY_SCHEMA,
        command.RESULT_FILENAME: command.RESULT_SCHEMA,
    }
    assert command.COMMAND_ID.endswith("-v8")
    for filename, schema in expected_schemas.items():
        assert json.loads((root / filename).read_text("utf-8"))["schema"] == schema
    for filename in (
        command.RANK_RESULT_FILENAME,
        command.FORMULA_FREEZE_FILENAME,
        command.REPLAY_FILENAME,
        command.RESULT_FILENAME,
    ):
        assert json.loads((root / filename).read_text("utf-8"))[
            "typed_gap_status"
        ] is None
    semantic_input = json.loads(
        (root / command.SEMANTIC_PROPOSAL_INPUT_FILENAME).read_text("utf-8")
    )
    assert semantic_input["named_image_count"] == 24
    assert [item["name"] for item in semantic_input["named_image_commitments"]] == [
        name
        for index in range(command.PANEL_COUNT)
        for name in (
            f"panel_{index:03d}.png",
            f"panel_{index:03d}_objects_000.png",
        )
    ]
    assert semantic_input["prepared_input"][
        "pixels_or_images_in_proposer_input"
    ] is True
    semantic_manifest = json.loads(
        (
            root
            / command.JOURNAL_DIRECTORY
            / "semantic_registry_proposer"
            / "manifest.json"
        ).read_text("utf-8")
    )
    assert semantic_manifest["modality"] == "named_image_structured"
    assert semantic_manifest["named_images"] == semantic_input[
        "named_image_commitments"
    ]
    semantic_result = json.loads(
        (root / command.SEMANTIC_PROPOSAL_RESULT_FILENAME).read_text("utf-8")
    )
    registry_record = json.loads(
        (root / command.REGISTRY_FREEZE_FILENAME).read_text("utf-8")
    )
    registry_tags = registry_record["registry"]["tags"]
    manifest = registry_record["registry_orientation_manifest"]
    assert manifest == [
        {
            "tag_id": item["tag_id"],
            "tag_digest": item["tag_digest"],
            "orientation_constraint": item["orientation_constraint"],
        }
        for item in registry_tags
    ]
    assert [item["orientation_constraint"] for item in manifest] == [
        "group0_positive",
        "group1_positive",
    ]
    assert registry_record["registry_orientation_manifest_digest"] == canonical_digest(
        {
            "schema": (
                "gkm.bongard-scene-predicate-registry-orientation-manifest.v1"
            ),
            "rows": manifest,
            "orientation_is_part_of_tag_digest": True,
        }
    )
    concepts = [
        *semantic_result["semantic_proposal"]["side0_positive"],
        *semantic_result["semantic_proposal"]["side1_positive"],
    ]
    assert all(concept["required_witnesses"] for concept in concepts)
    assert all("accepted_variants" in concept for concept in concepts)
    assert all("near_miss_boundaries" in concept for concept in concepts)
    bindings = semantic_input["prepared_input"]["model_view"][
        "required_positive_binding_panels"
    ]
    for orientation in ("side0_positive", "side1_positive"):
        assert all(
            [
                row["panel_alias"]
                for row in concept["support_bindings"]
            ]
            == bindings[orientation]
            for concept in semantic_result["semantic_proposal"][orientation]
        )
        assert all(
            {row["target_alias"] for row in concept["support_bindings"]}
            == {"whole_panel"}
            for concept in semantic_result["semantic_proposal"][orientation]
        )
    assert len(ranker_calls) == 1
    assert len(proposer_calls) == 1
    prompt = ranker_calls[0]
    for hidden in ("3" * 64, "4" * 64, "5" * 64):
        assert hidden not in prompt
    assert verified.selected_survivor_digest in prompt
    before = (len(visual_calls), len(proposer_calls), len(ranker_calls))
    replayed = command.verify_object_bongard_scene_predicate_calibration(
        root, source_root="offline-source"
    )
    assert replayed == verified
    assert (len(visual_calls), len(proposer_calls), len(ranker_calls)) == before


def test_real_ir_builds_readable_digest_free_ranker_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "real_ir_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root, monkeypatch, calibration_inputs, accepted=None
    )

    assert verified.status == "accepted"
    assert len(visual_calls) == 36
    assert len(proposer_calls) == 1
    assert len(ranker_calls) == 1
    assessment = json.loads((root / command.ASSESSMENT_FILENAME).read_text("utf-8"))
    bundle = assessment["ir_bundle"]
    rank_input = json.loads(
        (root / command.RANK_INPUT_FREEZE_FILENAME).read_text("utf-8")
    )
    ranker_slate = rank_input["ranker_slate"]
    assert bundle["complete_survivor_digests"]
    assert len(ranker_slate) <= 64
    assert {
        row["candidate_digest"] for row in ranker_slate
    } | {
        row["candidate_digest"] for row in bundle["omitted_survivors"]
    } == set(bundle["complete_survivor_digests"])
    survivor_complexities = {
        row["candidate_digest"]: row["complexity"]
        for row in bundle["candidates"]
        if row["candidate_digest"] in bundle["complete_survivor_digests"]
    }
    minimum_complexity = min(survivor_complexities.values())
    if (
        len(ranker_slate) < 64
        and any(value > minimum_complexity for value in survivor_complexities.values())
    ):
        assert any(
            row["complexity"] > minimum_complexity
            for row in ranker_slate
        )
    visible = json.dumps(ranker_slate, sort_keys=True)
    assert "bird-like object" in visible
    assert "a compact body has two wing-like extensions" in visible
    # A zero/upper-bounded count of a positive soft predicate is merely NOT
    # EXISTS written sideways.  It must never manufacture the reverse
    # orientation as a "positive" survivor.
    assert {
        row["orientation"] for row in ranker_slate
    } == {"group0_positive"}

    def count_comparisons(value: object) -> list[dict[str, object]]:
        found: list[dict[str, object]] = []
        if isinstance(value, dict):
            comparison = value.get("count_comparison")
            if isinstance(comparison, dict):
                found.append(comparison)
            for child in value.values():
                found.extend(count_comparisons(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(count_comparisons(child))
        return found

    assert all(
        item["comparison"] in {"at_least", "equal"} and item["value"] >= 1
        for item in count_comparisons(ranker_slate)
    )

    def digest_paths(value: object, path: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if (
                    isinstance(child, str)
                    and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", child)
                ):
                    found.append(child_path)
                found.extend(digest_paths(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(digest_paths(child, f"{path}[{index}]"))
        return found

    assert all(
        path.endswith(".candidate_digest")
        or path == "candidate_digest"
        for path in digest_paths(ranker_slate)
    )


def test_repeatability_failure_is_typed_grounding_gap_and_never_calls_ranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "gap_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root, monkeypatch, calibration_inputs, accepted=False
    )

    assert verified.status == "typed_grounding_repeatability_gap"
    assert verified.selected_survivor_digest is None
    assert verified.visual_fresh_call_count == 36
    assert verified.semantic_proposer_fresh_call_count == 1
    assert verified.ranker_fresh_call_count == 0
    result = json.loads((root / command.RESULT_FILENAME).read_text("utf-8"))
    assert result["physical_model_call_count"] == 37
    for filename in (
        command.RANK_RESULT_FILENAME,
        command.FORMULA_FREEZE_FILENAME,
        command.REPLAY_FILENAME,
        command.RESULT_FILENAME,
    ):
        assert json.loads((root / filename).read_text("utf-8"))[
            "typed_gap_status"
        ] == "typed_grounding_repeatability_gap"
    assert ranker_calls == []
    assert len(proposer_calls) == 1
    assert not (root / command.JOURNAL_DIRECTORY / "ranker").exists()
    before = len(visual_calls)
    replayed = command.verify_object_bongard_scene_predicate_calibration(
        root, source_root="offline-source"
    )
    assert replayed == verified
    assert len(visual_calls) == before


def test_semantic_invalid_payload_is_typed_gap_and_never_calls_ranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "semantic_gap_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root,
        monkeypatch,
        calibration_inputs,
        accepted=True,
        semantic_valid=False,
    )

    assert verified.status == "typed_semantic_proposal_gap"
    assert verified.selected_survivor_digest is None
    assert verified.visual_fresh_call_count == 36
    assert verified.semantic_proposer_fresh_call_count == 1
    assert verified.ranker_fresh_call_count == 0
    result = json.loads((root / command.RESULT_FILENAME).read_text("utf-8"))
    assert result["physical_model_call_count"] == 37
    for filename in (
        command.RANK_RESULT_FILENAME,
        command.FORMULA_FREEZE_FILENAME,
        command.REPLAY_FILENAME,
        command.RESULT_FILENAME,
    ):
        assert json.loads((root / filename).read_text("utf-8"))[
            "typed_gap_status"
        ] == "typed_semantic_proposal_gap"
    assert len(proposer_calls) == 1
    assert ranker_calls == []
    assert not (root / command.JOURNAL_DIRECTORY / "ranker").exists()
    registry = json.loads(
        (root / command.REGISTRY_FREEZE_FILENAME).read_text("utf-8")
    )["registry"]
    assert registry["tags"] == []
    before = (len(visual_calls), len(proposer_calls), len(ranker_calls))
    replayed = command.verify_object_bongard_scene_predicate_calibration(
        root, source_root="offline-source"
    )
    assert replayed == verified
    assert (len(visual_calls), len(proposer_calls), len(ranker_calls)) == before


@pytest.mark.parametrize(
    (
        "semantic_valid",
        "accepted",
        "coverage_passed",
        "selectivity_passed",
        "repeatability_passed",
        "expected",
    ),
    (
        (False, True, True, True, True, "typed_semantic_proposal_gap"),
        (True, False, False, True, True, "typed_language_gap"),
        (True, False, True, False, True, "typed_selectivity_gap"),
        (
            True,
            False,
            True,
            True,
            False,
            "typed_grounding_repeatability_gap",
        ),
        (True, True, True, True, True, None),
    ),
)
def test_typed_gap_status_names_the_failed_evidence_stage(
    semantic_valid: bool,
    accepted: bool,
    coverage_passed: bool,
    selectivity_passed: bool,
    repeatability_passed: bool,
    expected: str | None,
) -> None:
    registry = SimpleNamespace(registry_digest="a" * 64)
    proposal = SimpleNamespace(proposal_digest="b" * 64)
    bundle = json.loads(json.dumps(_bundle(registry, proposal, accepted=accepted)))
    bundle["coverage_gate"]["passed"] = coverage_passed
    bundle["selectivity_gate"]["passed"] = selectivity_passed
    bundle["repeatability_gate"]["passed"] = repeatability_passed
    bundle = _redigest_bundle(bundle)

    assert command._typed_calibration_gap_status(
        semantic_proposal_valid=semantic_valid,
        ir_bundle=bundle,
    ) == expected


def test_exact_frequency_registry_cannot_authorize_benchmark_acceptance() -> None:
    registry = SimpleNamespace(registry_digest="a" * 64)
    proposal = SimpleNamespace(proposal_digest="b" * 64)
    bundle = _bundle(registry, proposal, accepted=True)
    bundle["registry_derivation_mode"] = "exact_open_tag_frequency"
    bundle = _redigest_bundle(bundle)
    with pytest.raises(
        command.ObjectBongardScenePredicateCalibrationCommandError,
        match="registry derivation mode differs",
    ):
        command._validate_ir_bundle(bundle)

    exact_frequency = {
        "semantic_proposal_valid": True,
        "registry_derivation_mode": "exact_open_tag_frequency",
        "benchmark_acceptance_authorized_registry": True,
    }
    with pytest.raises(
        command.ObjectBongardScenePredicateCalibrationCommandError,
        match="acceptance lacks a role-aware semantic registry",
    ):
        command._result_record(
            inputs=SimpleNamespace(),
            authorization={},
            precommit={},
            discovery_batch={},
            discovery_freeze={},
            semantic_proposal_input={},
            semantic_proposal_result=exact_frequency,
            registry_record=exact_frequency,
            evaluation_a_batch={},
            evaluation_b_batch={},
            evaluation_freeze={},
            role_reveal={},
            assessment=exact_frequency,
            rank_input={},
            rank_result={},
            formula_freeze={
                "status": "accepted",
                "benchmark_acceptance_authorized_registry": True,
            },
            replay={},
        )


def test_semantic_optional_bad_row_is_quarantined_and_valid_pipeline_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "semantic_quarantine_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root,
        monkeypatch,
        calibration_inputs,
        accepted=True,
        semantic_extra_invalid=True,
    )

    assert verified.status == "accepted"
    assert len(visual_calls) == 36
    assert len(proposer_calls) == 1
    assert len(ranker_calls) == 1
    semantic_result = json.loads(
        (root / command.SEMANTIC_PROPOSAL_RESULT_FILENAME).read_text("utf-8")
    )
    assert semantic_result["semantic_proposal_status"] == "proposed"
    assert semantic_result["semantic_proposal_valid"] is True
    assert [
        row["reason_code"]
        for row in semantic_result["semantic_proposal"]["dropped_concepts"]
    ] == ["phrase_policy"]
    registry = json.loads(
        (root / command.REGISTRY_FREEZE_FILENAME).read_text("utf-8")
    )["registry"]
    assert {row["tag"] for row in registry["tags"]} == {
        "bird-like object",
        "three-sided frame with distinct edge markers",
    }
    assert command.verify_object_bongard_scene_predicate_calibration(
        root, source_root="offline-source"
    ) == verified


def test_ranker_privacy_rejects_formula_digest_leak(
    calibration_inputs: command._CalibrationInputs,
) -> None:
    candidate_digest = "8" * 64
    leaked_formula_digest = "9" * 64
    prompt = command._ranker_prompt(
        (
            {
                "candidate_digest": candidate_digest,
                "formula_digest": leaked_formula_digest,
                "formula": {"node": "atom", "kind": "registered_tag"},
            },
        )
    )
    with pytest.raises(
        command.ObjectBongardScenePredicateCalibrationCommandError,
        match="ranker prompt leaks",
    ):
        command._assert_ranker_privacy(
            prompt,
            inputs=calibration_inputs,
            hidden_digests=(leaked_formula_digest,),
        )


def test_command_is_python_canonical_and_has_no_lean_or_legacy_observer_import() -> None:
    from bongard.object_bongard_scene_predicate_ir import (
        SCENE_CALIBRATION_BUNDLE_SCHEMA,
    )

    tree = ast.parse(Path(command.__file__).read_text("utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("lean" in value.lower() for value in imports)
    assert not any("object_bongard_observer" in value for value in imports)
    authority = command._authority_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["frozen_python_predicate_is_normative"] is True
    assert authority["python_replay_is_normative"] is True
    assert authority["semantic_proposal_orientation_is_part_of_tag_identity"] is True
    assert authority["same_semantic_tag_tried_in_both_orientations"] is False
    assert authority[
        "registered_evaluator_receives_orientation_constraint_metadata"
    ] is False
    assert authority[
        "opposite_orientation_registered_tag_candidate_copies_forbidden"
    ] is True
    assert authority["lean_present"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert authority["lean_if_present_is_optional_checker_or_export_only"] is True
    assert authority["lean_affects_acceptance_or_runtime_semantics"] is False
    assert command.IR_BUNDLE_SCHEMA == SCENE_CALIBRATION_BUNDLE_SCHEMA
