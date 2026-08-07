"""Offline adversarial tests for the pinned Codex no-tools boundary."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest

import bongard.codex_no_tools_preflight as P
import bongard.transport as T


def _request(modality: str) -> dict[str, object]:
    content: list[dict[str, str]] = []
    if modality == "named_image":
        content.extend((
            {
                "type": "input_text",
                "text": '<image name=[Image #1] path="/private/synthetic.png">',
            },
            {
                "type": "input_image",
                "image_url": (
                    "data:image/png;base64,"
                    + base64.b64encode(P._CAPTURE_PNG).decode("ascii")),
            },
            {"type": "input_text", "text": "</image>"},
        ))
    content.append({"type": "input_text", "text": P._CAPTURE_PROMPT})
    return {
        "model": T.DEFAULT_CODEX_MODEL,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": []},
            {"type": "message", "role": "user", "content": content},
        ],
        "text": {
            "format": {
                "name": "codex_output_schema",
                "schema": P._CAPTURE_SCHEMA,
                "strict": True,
                "type": "json_schema",
            },
            "verbosity": "low",
        },
    }


def _row(modality: str) -> dict[str, object]:
    request = _request(modality)
    raw = T._canonical_json_bytes(request)
    return {
        "schema": P._CAPTURE_ROW_SCHEMA,
        "modality": modality,
        "normalized_command_digest": T._codex_command_digest(
            P._normalized_capture_command(modality)),
        "fixture_prompt_digest": T._bytes_digest(
            P._CAPTURE_PROMPT.encode("utf-8")),
        "fixture_schema_digest": T._bytes_digest(
            T._canonical_json_bytes(P._CAPTURE_SCHEMA)),
        "fixture_image_digest": (
            "" if modality == "text" else T._bytes_digest(P._CAPTURE_PNG)),
        "request_method": "POST",
        "request_path": "/v1/responses",
        "request_content_encoding": "absent",
        "request_body_raw_base64": base64.b64encode(raw).decode("ascii"),
        "request_body_raw_digest": T._bytes_digest(raw),
        "request_body_canonical_digest": T._digest(request),
        "model": T.DEFAULT_CODEX_MODEL,
        "responses_lite": True,
        "top_level_tools_absent": True,
        "additional_tools_count": 1,
        "tool_surface_digest": T.CODEX_TOOL_SURFACE_DIGEST,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "outcome": "success",
    }


def _attestation() -> P.CodexNoToolsAttestation:
    body: dict[str, object] = {
        "schema": P.CODEX_NO_TOOLS_ATTESTATION_SCHEMA,
        "launcher_version": T.PINNED_CODEX_CLI_VERSION,
        "launcher_digest": "b" * 64,
        "capture_harness_source_digest": P._LOADED_SOURCE_SHA256,
        "model_catalog_digest": T.PINNED_MODEL_CATALOG_RAW_DIGEST,
        "model_catalog_canonical_digest": (
            T.PINNED_MODEL_CATALOG_CANONICAL_DIGEST),
        "bundled_catalog_raw_digest": T.PINNED_BUNDLED_CATALOG_RAW_DIGEST,
        "bundled_model_record_canonical_digest": (
            T.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST),
        "catalog_diff_policy": (
            "only-tool_mode-direct-and-apply_patch_tool_type-null"),
        "cloud_config_bundle_cache_binding": "absent",
        "transport_policy_digest": T.CODEX_TRANSPORT_POLICY_DIGEST,
        "effective_tool_mode": T.CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": T.CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": T.CODEX_TOOL_SURFACE_DIGEST,
        "modality_coverage": ["text", "named_image"],
        "captures": [_row("text"), _row("named_image")],
        "outcome": "success",
    }
    body["attestation_digest"] = T._digest(body)
    return P.CodexNoToolsAttestation.from_mapping(body)


def _redigest(value: dict[str, object]) -> None:
    value["attestation_digest"] = T._digest({
        key: item for key, item in value.items()
        if key != "attestation_digest"
    })


def test_pinned_catalog_is_exact_one_model_two_field_delta() -> None:
    snapshot = T.snapshot_pinned_model_catalog()
    decoded = json.loads(snapshot.data)
    assert snapshot.raw_digest == T.PINNED_MODEL_CATALOG_RAW_DIGEST
    assert snapshot.canonical_digest == T.PINNED_MODEL_CATALOG_CANONICAL_DIGEST
    assert len(decoded["models"]) == 1
    record = decoded["models"][0]
    assert record["slug"] == "gpt-5.6-sol"
    assert record["tool_mode"] == "direct"
    assert record["apply_patch_tool_type"] is None
    source = dict(record)
    source["tool_mode"] = "code_mode_only"
    source["apply_patch_tool_type"] = "freeform"
    assert T._digest(source) == T.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST


def test_exact_command_pins_catalog_and_all_tool_gates(tmp_path: Path) -> None:
    command = T._codex_command(
        executable="/pinned/codex",
        view_dir=str(tmp_path),
        image_paths=(),
        schema_path=str(tmp_path / "output_schema.json"),
        model_catalog_path=str(tmp_path / "model_catalog.json"),
        model=T.DEFAULT_CODEX_MODEL,
        reasoning_effort="medium",
    )
    configs = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--config"
    ]
    assert configs == [
        "model_catalog_json=" + json.dumps(str(tmp_path / "model_catalog.json")),
        'model_reasoning_effort="medium"',
        *T._STRICT_CONFIG_OVERRIDES,
    ]
    disabled = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--disable"
    ]
    assert disabled == list(T._DISABLED_FEATURES)
    assert "code_mode" in disabled
    assert "code_mode_host" in disabled
    assert "enable_request_compression" in disabled
    assert "tools.update_plan.enabled=false" in configs
    assert "tools.experimental_request_user_input.enabled=false" in configs


def test_staged_catalog_detects_byte_and_metadata_toctou(tmp_path: Path) -> None:
    stage = T._stage_model_catalog(
        str(tmp_path), T.snapshot_pinned_model_catalog())
    T._recheck_staged_model_catalog(stage)
    path = Path(stage.path)
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(T.CodexProposerFailure, match="metadata changed"):
        T._recheck_staged_model_catalog(stage)


def test_attestation_reparses_bodies_and_rejects_tool_or_modality_forgery() -> None:
    attestation = _attestation()
    assert P.validate_codex_no_tools_attestation(
        attestation,
        expected_launcher_digest="b" * 64,
        expected_model_catalog_digest=T.PINNED_MODEL_CATALOG_RAW_DIGEST,
        expected_cloud_policy_cache_binding="absent",
    ) == attestation

    forged = copy.deepcopy(attestation.to_dict())
    row = forged["captures"][0]
    request = json.loads(base64.b64decode(row["request_body_raw_base64"]))
    request["input"][0]["tools"] = [{"type": "function", "name": "escape"}]
    raw = T._canonical_json_bytes(request)
    row["request_body_raw_base64"] = base64.b64encode(raw).decode("ascii")
    row["request_body_raw_digest"] = T._bytes_digest(raw)
    row["request_body_canonical_digest"] = T._digest(request)
    _redigest(forged)
    with pytest.raises(T.CodexProposerFailure, match="additional-tools"):
        P.CodexNoToolsAttestation.from_mapping(forged)

    relabelled = copy.deepcopy(attestation.to_dict())
    relabelled["captures"][1] = copy.deepcopy(relabelled["captures"][0])
    relabelled["captures"][1]["modality"] = "named_image"
    relabelled["captures"][1]["fixture_image_digest"] = T._bytes_digest(
        P._CAPTURE_PNG)
    relabelled["captures"][1]["normalized_command_digest"] = (
        T._codex_command_digest(P._normalized_capture_command("named_image")))
    _redigest(relabelled)
    with pytest.raises(T.CodexProposerFailure, match="synthetic PNG"):
        P.CodexNoToolsAttestation.from_mapping(relabelled)


def test_attestation_rejects_redigested_command_template_substitution() -> None:
    forged = _attestation().to_dict()
    forged["captures"][0]["normalized_command_digest"] = "c" * 64
    _redigest(forged)
    with pytest.raises(T.CodexProposerFailure, match="capture row policy"):
        P.CodexNoToolsAttestation.from_mapping(forged)


@pytest.mark.parametrize(
    "mutation",
    ("top_level", "missing_additional", "duplicate_additional", "prompt"),
)
def test_attestation_rejects_redigested_request_body_policy_mutations(
    mutation: str,
) -> None:
    forged = _attestation().to_dict()
    row = forged["captures"][0]
    request = json.loads(base64.b64decode(row["request_body_raw_base64"]))
    if mutation == "top_level":
        request["tools"] = []
    elif mutation == "missing_additional":
        request["input"].pop(0)
    elif mutation == "duplicate_additional":
        request["input"].insert(1, copy.deepcopy(request["input"][0]))
    else:
        request["input"][1]["content"][-1]["text"] += " altered"
    raw = T._canonical_json_bytes(request)
    row["request_body_raw_base64"] = base64.b64encode(raw).decode("ascii")
    row["request_body_raw_digest"] = T._bytes_digest(raw)
    row["request_body_canonical_digest"] = T._digest(request)
    _redigest(forged)
    with pytest.raises(T.CodexProposerFailure):
        P.CodexNoToolsAttestation.from_mapping(forged)


def test_attestation_expected_bindings_and_digest_are_fail_closed() -> None:
    attestation = _attestation()
    for kwargs in (
        {"expected_launcher_digest": "0" * 64},
        {"expected_model_catalog_digest": "0" * 64},
        {"expected_cloud_policy_cache_binding": "sha256:" + "0" * 64},
    ):
        with pytest.raises(T.CodexProposerFailure, match="differs"):
            P.validate_codex_no_tools_attestation(attestation, **kwargs)
    tampered = attestation.to_dict()
    tampered["attestation_digest"] = "0" * 64
    with pytest.raises(T.CodexProposerFailure, match="does not reproduce"):
        P.CodexNoToolsAttestation.from_mapping(tampered)


def test_capture_policy_disables_retries_and_rejects_zero_or_two_requests() -> None:
    configs = [
        command[index + 1]
        for command in (
            P._normalized_capture_command("text"),
            P._normalized_capture_command("named_image"),
        )
        for index, item in enumerate(command[:-1])
        if item == "--config"
    ]
    assert "model_providers.bongard_capture.request_max_retries=0" in configs
    assert "model_providers.bongard_capture.stream_max_retries=0" in configs
    assert "model_providers.bongard_capture.supports_websockets=false" in configs
    with pytest.raises(T.CodexProposerFailure, match="exactly one"):
        P._require_exactly_one_capture([])
    with pytest.raises(T.CodexProposerFailure, match="exactly one"):
        P._require_exactly_one_capture([{}, {}])
    assert P._require_exactly_one_capture([{"body": b"x"}]) == {
        "body": b"x"
    }


def test_transport_reuses_injected_attestation_without_recapture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation()

    def forbidden_capture(**_kwargs: object) -> object:
        raise AssertionError("a campaign-frozen attestation must not be recaptured")

    monkeypatch.setattr(P, "attest_codex_no_tools", forbidden_capture)
    digest = T._resolve_no_tools_attestation(
        executable="/not-executed",
        launcher_digest="b" * 64,
        model_catalog_snapshot=T.snapshot_pinned_model_catalog(),
        cloud_policy_cache_snapshot=T.CloudPolicyCacheSnapshot(None),
        attestation=attestation,
        expected_digest=attestation.attestation_digest,
    )
    assert digest == attestation.attestation_digest
