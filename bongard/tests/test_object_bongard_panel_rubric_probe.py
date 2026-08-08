"""Offline end-to-end test for the historical-only whole-panel probe."""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import bongard.transport as transport_module
from bongard.canonical import canonical_digest
from bongard.crack_lab.object_bongard_panel_rubric_probe import (
    DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
    DEFAULT_V10_NOMINATION_ROOT,
    ObjectBongardPanelRubricProbeError,
    PROBE_STATUS,
    _load_probe_inputs,
    _run_loaded_probe,
    _verify_loaded_probe,
)
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
) -> None:
    inputs = _load_probe_inputs(
        nomination_root=DEFAULT_V10_NOMINATION_ROOT,
        rejected_calibration_root=DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
        source_directory=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    )
    group_a = {item.png_sha256 for item in inputs.source.group_a_panels}
    group_b = {item.png_sha256 for item in inputs.source.group_b_panels}
    allowed = group_a | group_b
    lock = threading.Lock()
    calls: list[str] = []

    def transport(prompt, paths, names, schema, **kwargs):
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
                inputs.precommit.runtime,
            ),
        )

    root = tmp_path / "probe"
    launched = _run_loaded_probe(
        root, inputs, parallel_workers=4, transport=transport
    )
    assert launched.exact_survivor is True
    assert len(calls) == 12 and set(calls) == allowed
    assert len(tuple((root / "artifacts").iterdir())) == 12
    assert len(tuple((root / "replays").iterdir())) == 12
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
        _run_loaded_probe(root, inputs, parallel_workers=4, transport=transport)
    assert len(calls) == 12
