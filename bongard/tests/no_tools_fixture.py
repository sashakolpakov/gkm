"""Offline canonical fixture for the frozen Codex no-tools campaign boundary."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard import codex_no_tools_preflight as preflight
from bongard import transport
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.transport import CodexModelCatalogSnapshot, CodexReceipt


def canonical_no_tools_runtime(
    launcher_digest: str,
    *,
    cloud_policy_cache_binding: str = "absent",
) -> tuple[CodexModelCatalogSnapshot, CodexNoToolsAttestation]:
    """Build valid synthetic capture evidence without executing Codex."""

    catalog = transport.snapshot_pinned_model_catalog()
    prompt_digest = hashlib.sha256(
        preflight._CAPTURE_PROMPT.encode("utf-8")
    ).hexdigest()
    schema_digest = hashlib.sha256(
        transport._canonical_json_bytes(preflight._CAPTURE_SCHEMA)
    ).hexdigest()
    captures: list[dict[str, object]] = []
    for modality in ("text", "named_image"):
        content: list[dict[str, str]] = []
        if modality == "named_image":
            content.extend(
                (
                    {
                        "type": "input_text",
                        "text": (
                            '<image name=[Image #1] '
                            'path="/private/synthetic.png">'
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + base64.b64encode(preflight._CAPTURE_PNG).decode(
                                "ascii"
                            )
                        ),
                    },
                    {"type": "input_text", "text": "</image>"},
                )
            )
        content.append(
            {"type": "input_text", "text": preflight._CAPTURE_PROMPT}
        )
        request: dict[str, object] = {
            "model": transport.DEFAULT_CODEX_MODEL,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": []},
                {"type": "message", "role": "user", "content": content},
            ],
            "text": {
                "format": {
                    "name": "codex_output_schema",
                    "schema": preflight._CAPTURE_SCHEMA,
                    "strict": True,
                    "type": "json_schema",
                },
                "verbosity": "low",
            },
        }
        raw_request = transport._canonical_json_bytes(request)
        captures.append(
            {
                "schema": preflight._CAPTURE_ROW_SCHEMA,
                "modality": modality,
                "normalized_command_digest": transport._codex_command_digest(
                    preflight._normalized_capture_command(modality)
                ),
                "fixture_prompt_digest": prompt_digest,
                "fixture_schema_digest": schema_digest,
                "fixture_image_digest": (
                    ""
                    if modality == "text"
                    else hashlib.sha256(preflight._CAPTURE_PNG).hexdigest()
                ),
                "request_method": "POST",
                "request_path": "/v1/responses",
                "request_content_encoding": "absent",
                "request_body_raw_base64": base64.b64encode(raw_request).decode(
                    "ascii"
                ),
                "request_body_raw_digest": hashlib.sha256(raw_request).hexdigest(),
                "request_body_canonical_digest": transport._digest(request),
                "model": transport.DEFAULT_CODEX_MODEL,
                "responses_lite": True,
                "top_level_tools_absent": True,
                "additional_tools_count": 1,
                "tool_surface_digest": transport.CODEX_TOOL_SURFACE_DIGEST,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "outcome": "success",
            }
        )
    body: dict[str, object] = {
        "schema": preflight.CODEX_NO_TOOLS_ATTESTATION_SCHEMA,
        "launcher_version": transport.PINNED_CODEX_CLI_VERSION,
        "launcher_digest": launcher_digest,
        "capture_harness_source_digest": preflight._LOADED_SOURCE_SHA256,
        "model_catalog_digest": catalog.raw_digest,
        "model_catalog_canonical_digest": catalog.canonical_digest,
        "bundled_catalog_raw_digest": transport.PINNED_BUNDLED_CATALOG_RAW_DIGEST,
        "bundled_model_record_canonical_digest": (
            transport.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST
        ),
        "catalog_diff_policy": (
            "only-tool_mode-direct-and-apply_patch_tool_type-null"
        ),
        "cloud_config_bundle_cache_binding": cloud_policy_cache_binding,
        "transport_policy_digest": transport.CODEX_TRANSPORT_POLICY_DIGEST,
        "effective_tool_mode": transport.CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": transport.CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": transport.CODEX_TOOL_SURFACE_DIGEST,
        "modality_coverage": ["text", "named_image"],
        "captures": captures,
        "outcome": "success",
    }
    record = {**body, "attestation_digest": transport._digest(body)}
    return catalog, CodexNoToolsAttestation.from_mapping(record)


def canonical_codex_receipt(
    prompt: str,
    paths: Sequence[str],
    output_schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    launcher_digest: str,
    reasoning_effort: str,
    model: str = transport.DEFAULT_CODEX_MODEL,
    names: Sequence[str] | None = None,
    command_fixture: str = "offline canonical no-tools turn",
) -> CodexReceipt:
    """Create and validate one exact pinned receipt-v4 test fixture."""

    catalog, attestation = canonical_no_tools_runtime(launcher_digest)
    path_values = tuple(paths)
    name_values = (
        tuple(Path(path).name for path in path_values)
        if names is None
        else tuple(names)
    )
    if len(path_values) != len(name_values):
        raise ValueError("fixture paths and names differ in length")
    identities = [
        {
            "name": name,
            "byte_count": len(data),
            "content_digest": hashlib.sha256(data).hexdigest(),
        }
        for path, name in zip(path_values, name_values, strict=True)
        for data in (Path(path).read_bytes(),)
    ]
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = transport._digest(dict(output_schema))
    view_digest = transport._digest(identities)
    binding = {
        "model_catalog_digest": catalog.raw_digest,
        "transport_policy_digest": transport.CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": transport._digest(
            {
                "fixture": command_fixture,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "modality": "structured" if names is None else "named_image",
            }
        ),
        "effective_tool_mode": transport.CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": transport.CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": transport.CODEX_TOOL_SURFACE_DIGEST,
        "tool_surface_attestation_digest": attestation.attestation_digest,
    }
    if names is None:
        input_schema = transport.STRUCTURED_INPUT_DIGEST_SCHEMA
        set_digest = transport.semantic_panel_set_digest(path_values)
        envelope = {
            "schema": input_schema,
            "task": prompt,
            "ordered_panel_identities": identities,
            "panel_view_digest": view_digest,
            "panel_set_digest": set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
            "transport": binding,
        }
    else:
        input_schema = transport.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
        set_digest = transport.named_image_set_digest(path_values, name_values)
        envelope = {
            "schema": input_schema,
            "task": prompt,
            "ordered_image_identities": identities,
            "image_view_digest": view_digest,
            "image_set_digest": set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
            "transport": binding,
        }
    body: dict[str, Any] = {
        "schema": transport.CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": model,
        "model_identity_evidence": "jsonl-reported-model",
        "requested_reasoning_effort": reasoning_effort,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": "00000000-0000-4000-8000-000000000001",
        "codex_cli_version": transport.PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": launcher_digest,
        "cloud_config_bundle_cache_binding": "absent",
        **binding,
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": input_schema,
        "input_digest": transport._digest(envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
        "structured_output_digest": transport._digest(dict(payload)),
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
        "isolation_policy": transport.CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = transport._digest(body)
    transport.validate_codex_receipt(body)
    receipt = CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )
    if names is not None:
        transport.validate_codex_named_image_receipt(
            receipt,
            prompt,
            path_values,
            name_values,
            output_schema,
            payload,
        )
    return receipt


__all__ = ["canonical_codex_receipt", "canonical_no_tools_runtime"]
