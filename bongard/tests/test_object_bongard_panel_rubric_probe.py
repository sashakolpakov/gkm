"""Offline end-to-end test for the historical-only whole-panel probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import bongard.transport as transport_module
import bongard.object_bongard_rubric_calibration as calibration_module
from bongard.canonical import canonical_digest
from bongard.crack_lab.object_bongard_panel_rubric_probe import (
    DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
    DEFAULT_V10_NOMINATION_ROOT,
    DIAGNOSTIC_EXPECTED_LAUNCHER_SHA256,
    DIAGNOSTIC_EXECUTABLE,
    DIAGNOSTIC_MINUTES,
    DIAGNOSTIC_MODEL,
    DIAGNOSTIC_REASONING_EFFORT,
    ObjectBongardPanelRubricProbeError,
    PROBE_STATUS,
    REJECTED_V10_CALIBRATION_SOURCE_DIGEST,
    V10_NOMINATION_SOURCE_DIGEST,
    _load_probe_inputs,
    _run_loaded_probe,
    _verify_loaded_probe,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
)
from bongard.transport import (
    CODEX_APPLY_PATCH_TOOL_TYPE,
    CODEX_EFFECTIVE_TOOL_MODE,
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    CODEX_TOOL_SURFACE_DIGEST,
    CODEX_TRANSPORT_POLICY_DIGEST,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    PINNED_CODEX_CLI_VERSION,
    CodexReceipt,
    CodexStructuredResult,
    CloudPolicyCacheSnapshot,
)
import pytest


def _receipt(
    prompt: str,
    paths: Sequence[str],
    names: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    runtime,
) -> CodexReceipt:
    identities = [
        {
            "name": name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path, name in zip(paths, names, strict=True)
    ]
    schema_digest = canonical_digest(dict(schema))
    view_digest = canonical_digest(identities)
    set_digest = "sha256:" + canonical_digest(
        {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": identities}
    )
    captures = runtime.no_tools_attestation.to_dict()["captures"]
    named_capture = next(item for item in captures if item["modality"] == "named_image")
    binding = {
        "model_catalog_digest": runtime.model_catalog_snapshot.raw_digest,
        "transport_policy_digest": CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": named_capture["normalized_command_digest"],
        "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        "tool_surface_attestation_digest": runtime.no_tools_attestation.attestation_digest,
    }
    causal = transport_module._causal_named_image_input_metadata(
        prompt,
        paths,
        names,
        schema_digest,
        view_digest,
        set_digest,
        binding,
    )
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": runtime.model,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": runtime.reasoning_effort,
        "input_tokens": 20,
        "cached_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_output_tokens": 2,
        "thread_id": "00000000-0000-4000-8000-000000000099",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": runtime.expected_launcher_digest,
        "cloud_config_bundle_cache_binding": runtime.policy_cache_binding,
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


def test_historical_only_probe_makes_twelve_blind_calls_and_cold_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def obsolete_source_reconstruction(*args, **kwargs):
        raise AssertionError("mutable calibration geometry must not be reconstructed")

    monkeypatch.setattr(
        calibration_module,
        "load_object_bongard_rubric_calibration_source",
        obsolete_source_reconstruction,
    )
    inputs = _load_probe_inputs(
        nomination_root=DEFAULT_V10_NOMINATION_ROOT,
        rejected_calibration_root=DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
        source_directory=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    )
    assert inputs.source.source_digest == V10_NOMINATION_SOURCE_DIGEST
    assert (
        inputs.source.rejected_calibration_source_digest
        == REJECTED_V10_CALIBRATION_SOURCE_DIGEST
    )
    assert inputs.nomination.artifact.receipt is not None
    assert inputs.nomination.artifact.model_payload == {
        "proposal_0": {
            "group_0_cue_text": (
                "One decorated figure forms two closed loops touching at one vertex"
            ),
            "group_1_cue_text": (
                "One decorated figure forms a closed loop with a dangling branch"
            ),
        },
        "proposal_1": {
            "group_0_cue_text": "One undecorated figure appears",
            "group_1_cue_text": "Outlined triangular beads decorate a figure",
        },
    }
    group_a = {item.png_sha256 for item in inputs.source.group_a_panels}
    group_b = {item.png_sha256 for item in inputs.source.group_b_panels}
    allowed = group_a | group_b
    fresh_cache = CloudPolicyCacheSnapshot(None)
    fresh_catalog, fresh_attestation = canonical_no_tools_runtime(
        DIAGNOSTIC_EXPECTED_LAUNCHER_SHA256
    )
    fresh_runtime = ObjectBongardTurnRuntime(
        model=DIAGNOSTIC_MODEL,
        reasoning_effort=DIAGNOSTIC_REASONING_EFFORT,
        minutes=DIAGNOSTIC_MINUTES,
        verbose=False,
        executable=DIAGNOSTIC_EXECUTABLE,
        cloud_policy_cache_snapshot=fresh_cache,
        model_catalog_snapshot=fresh_catalog,
        expected_launcher_digest=DIAGNOSTIC_EXPECTED_LAUNCHER_SHA256,
        no_tools_attestation=fresh_attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    preflight_calls: list[str] = []

    def cache_snapshotter():
        preflight_calls.append("cache")
        return fresh_cache

    def catalog_snapshotter():
        preflight_calls.append("catalog")
        return fresh_catalog

    def fingerprinter(executable, *, expected_launcher_digest):
        assert executable == DIAGNOSTIC_EXECUTABLE
        assert expected_launcher_digest == DIAGNOSTIC_EXPECTED_LAUNCHER_SHA256
        preflight_calls.append("fingerprint")
        return {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": expected_launcher_digest,
        }

    def attester(**kwargs):
        assert kwargs["cloud_policy_cache_snapshot"] is fresh_cache
        assert kwargs["model_catalog_snapshot"] is fresh_catalog
        preflight_calls.append("attestation")
        return fresh_attestation

    lock = threading.Lock()
    calls: list[str] = []

    def transport(prompt, paths, names, schema, **kwargs):
        assert (root / "diagnostic_authorization.json").is_file()
        assert (root / "diagnostic_execution_precommit.json").is_file()
        assert (root / "manifest.json").is_file()
        assert names == ("panel.png",)
        assert len(paths) == 1
        lowered = prompt.lower()
        assert "group_a" not in lowered and "group_b" not in lowered
        assert "query" not in lowered and "official" not in lowered
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        assert panel_digest in allowed
        with lock:
            calls.append(panel_digest)
        payload = (
            {"lower": 3, "upper": 4}
            if panel_digest in group_a
            else {"lower": 0, "upper": 1}
        )
        return CodexStructuredResult(
            payload,
            _receipt(
                prompt,
                paths,
                names,
                schema,
                payload,
                fresh_runtime,
            ),
        )

    root = tmp_path / "probe"
    launched = _run_loaded_probe(
        root,
        inputs,
        parallel_workers=4,
        transport=transport,
        cloud_policy_cache_snapshotter=cache_snapshotter,
        model_catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=fingerprinter,
        runtime_attester=attester,
    )
    assert preflight_calls == ["cache", "catalog", "fingerprint", "attestation"]
    assert fresh_runtime.binding != inputs.precommit.runtime.binding
    assert launched.exact_survivor is True
    assert len(calls) == 12 and set(calls) == allowed
    assert len(tuple((root / "artifacts").iterdir())) == 12
    assert len(tuple((root / "replays").iterdir())) == 12
    authorization = json.loads((root / "diagnostic_authorization.json").read_text())
    precommit = json.loads(
        (root / "diagnostic_execution_precommit.json").read_text()
    )
    assert authorization["runtime_policy"]["old_v10_runtime_objects_authorized"] is False
    assert precommit["old_v10_runtime_objects_reused"] is False
    assert precommit["fresh_runtime_snapshots_captured"] is True
    assert precommit["runtime_binding"] == fresh_runtime.binding
    assert launched.summary_data()["status"] == PROBE_STATUS
    assert (
        launched.summary_data()[
            "old_calibration_authorization_authorizes_probe_jobs"
        ]
        is False
    )

    replayed = _verify_loaded_probe(root, inputs)
    assert replayed == launched
    assert len(calls) == 12  # disk replay made no transport/model call

    with pytest.raises(ObjectBongardPanelRubricProbeError, match="fresh"):
        _run_loaded_probe(
            root,
            inputs,
            parallel_workers=4,
            transport=transport,
            cloud_policy_cache_snapshotter=cache_snapshotter,
            model_catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=fingerprinter,
            runtime_attester=attester,
        )
    assert len(calls) == 12

    failed_root = tmp_path / "failed_preflight"

    def failed_cache_snapshotter():
        raise RuntimeError("synthetic fresh-cache failure")

    with pytest.raises(RuntimeError, match="fresh-cache"):
        _run_loaded_probe(
            failed_root,
            inputs,
            parallel_workers=4,
            transport=transport,
            cloud_policy_cache_snapshotter=failed_cache_snapshotter,
            model_catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=fingerprinter,
            runtime_attester=attester,
        )
    assert (failed_root / "diagnostic_authorization.json").is_file()
    assert not (failed_root / "diagnostic_execution_precommit.json").exists()
    assert not (failed_root / "manifest.json").exists()
    assert len(calls) == 12
    with pytest.raises(ObjectBongardPanelRubricProbeError, match="fresh"):
        _run_loaded_probe(
            failed_root,
            inputs,
            parallel_workers=4,
            transport=transport,
            cloud_policy_cache_snapshotter=cache_snapshotter,
            model_catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=fingerprinter,
            runtime_attester=attester,
        )
    assert len(calls) == 12
