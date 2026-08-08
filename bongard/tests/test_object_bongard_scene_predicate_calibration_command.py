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
    panel_registered_tag_ids: tuple[str, ...] = (),
    entity_registered_tag_ids: tuple[str, ...] = (),
) -> dict[str, object]:
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
                            "state": "present" if role == 0 else "absent",
                            "evidence": "the frozen phrase was checked directly",
                        }
                        for tag_id in entity_registered_tag_ids
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
                        "state": "present" if role == 0 else "absent",
                        "evidence": "the frozen whole-panel phrase was checked directly",
                    }
                    for tag_id in panel_registered_tag_ids
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
    passed = accepted
    body: dict[str, object] = {
        "schema": command.IR_BUNDLE_SCHEMA,
        "ir_source_digest": "6" * 64,
        "algorithm_digest": "7" * 64,
        "registry_digest": registry.registry_digest,
        "registry_derivation_mode": "role_aware_semantic_concept_proposal",
        "registry_derivation_digest": semantic_registry_proposal.proposal_digest,
        "coverage_gate": {"passed": True, "covered_panel_count": 12},
        "selectivity_gate": {"passed": passed, "separated_panel_count": 12 if passed else 0},
        "repeatability_gate": {"passed": True, "repeat_tested_panel_count": 12},
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

    def named_transport(prompt, paths, names, schema, **_kwargs):
        assert (root / command.AUTHORIZATION_FILENAME).is_file()
        assert (root / command.PRECOMMIT_FILENAME).is_file()
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
        with lock:
            call_ordinal = visual_calls.count(stage)
            visual_calls.append(stage)
            if registered:
                registered_prompts[stage].append(prompt)
        payload = _visual_payload(
            inventory,
            registered=registered,
            role=0 if index < 6 else 1,
            marker=f"{stage}-{call_ordinal:02d}",
            panel_registered_tag_ids=tuple(
                schema["properties"]["panel"]["properties"]["registered_tags"][
                    "items"
                ]["properties"]["tag_id"].get("enum", ())
            ),
            entity_registered_tag_ids=tuple(
                schema["properties"]["objects"]["items"]["properties"]
                ["registered_tags"]["items"]["properties"]["tag_id"].get(
                    "enum", ()
                )
            ),
        )
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    def text_transport(prompt, schema, **_kwargs):
        properties = schema["properties"]
        if {"side0_positive", "side1_positive"}.issubset(properties):
            assert (root / command.DISCOVERY_FREEZE_FILENAME).is_file()
            assert (root / command.ROLE_REVEAL_FILENAME).is_file()
            assert not (root / command.REGISTRY_FREEZE_FILENAME).exists()
            proposer_calls.append(prompt)
            side0_aliases = properties["side0_positive"]["items"][
                "properties"
            ]["citations"]["items"]["enum"]
            side1_aliases = properties["side1_positive"]["items"][
                "properties"
            ]["citations"]["items"]["enum"]
            payload = {
                "side0_positive": [
                    {
                        "scope": "panel",
                        "phrase": "bird-like object",
                        "citations": side0_aliases[:2],
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
                        "citations": side1_aliases[:2],
                    }
                ],
            }
            if semantic_extra_invalid:
                payload["side0_positive"].append(
                    {
                        "scope": "entity",
                        "phrase": "pointed and curved",
                        "citations": side0_aliases[1:3],
                    }
                )
        else:
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
        assert "historical_role" not in prompt
        assert "side0_positive" not in prompt
        assert "side1_positive" not in prompt
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
    assert bundle["complete_survivor_digests"]
    assert len(bundle["ranker_slate"]) <= 64
    assert {
        row["candidate_digest"] for row in bundle["ranker_slate"]
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
        len(bundle["ranker_slate"]) < 64
        and any(value > minimum_complexity for value in survivor_complexities.values())
    ):
        assert any(
            row["complexity"] > minimum_complexity
            for row in bundle["ranker_slate"]
        )
    visible = json.dumps(bundle["ranker_slate"], sort_keys=True)
    assert "bird-like object" in visible
    assert "resembles a bird or flying bird silhouette" in visible
    # A zero/upper-bounded count of a positive soft predicate is merely NOT
    # EXISTS written sideways.  It must never manufacture the reverse
    # orientation as a "positive" survivor.
    assert {
        row["orientation"] for row in bundle["ranker_slate"]
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
        for item in count_comparisons(bundle["ranker_slate"])
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
        for path in digest_paths(bundle["ranker_slate"])
    )


def test_empty_survivor_is_typed_gap_and_never_calls_ranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_inputs: command._CalibrationInputs,
) -> None:
    root = tmp_path / "gap_scene_calibration"
    verified, visual_calls, proposer_calls, ranker_calls = _run(
        root, monkeypatch, calibration_inputs, accepted=False
    )

    assert verified.status == "typed_empty_survivor_gap"
    assert verified.selected_survivor_digest is None
    assert verified.visual_fresh_call_count == 36
    assert verified.semantic_proposer_fresh_call_count == 1
    assert verified.ranker_fresh_call_count == 0
    result = json.loads((root / command.RESULT_FILENAME).read_text("utf-8"))
    assert result["physical_model_call_count"] == 37
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
    assert authority["lean_present"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert command.IR_BUNDLE_SCHEMA == SCENE_CALIBRATION_BUNDLE_SCHEMA
