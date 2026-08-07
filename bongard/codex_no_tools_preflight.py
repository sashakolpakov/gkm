"""Installed-binary proof that pinned Codex exposes no callable tools.

The proof uses two synthetic, localhost-only Responses captures: one text-only
turn and one named-image turn.  It never sends corpus data or reaches a model
endpoint.  The server deliberately returns a non-retryable HTTP 400 after it
has captured the first request; request construction, not model output, is the
scientific fact certified here.
"""
from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
from typing import Any, Mapping, Sequence

import bongard.transport as T


CODEX_NO_TOOLS_ATTESTATION_SCHEMA = "bongard.codex-no-tools-attestation/v1"
_CAPTURE_ROW_SCHEMA = "bongard.codex-no-tools-capture-row/v1"
_CAPTURE_PROVIDER_ID = "bongard_capture"
_CAPTURE_PROMPT = (
    "Return the fixed synthetic value. This is a local transport preflight; "
    "do not infer or inspect any corpus data."
)
_CAPTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}
_CAPTURE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAWklEQVR4nJXS2woAIAgD"
    "0Lb//2eDBInKnD1URqcLDGY2Oo2t3WMHABoAa7di6EP8pDSMmWi4F4rhUZeG99Lf8HnMxzB7"
    "a2aYgcwU0bhNnaXDSOFz472a1rhnAmX8Kh1HC2CVAAAAAElFTkSuQmCC")
_MAX_CAPTURE_REQUEST_BYTES = 4_000_000
_MAX_ATTESTATION_BYTES = 12_000_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ATTESTATION_KEYS = frozenset({
    "schema",
    "launcher_version",
    "launcher_digest",
    "capture_harness_source_digest",
    "model_catalog_digest",
    "model_catalog_canonical_digest",
    "bundled_catalog_raw_digest",
    "bundled_model_record_canonical_digest",
    "catalog_diff_policy",
    "cloud_config_bundle_cache_binding",
    "transport_policy_digest",
    "effective_tool_mode",
    "apply_patch_tool_type",
    "tool_surface_digest",
    "modality_coverage",
    "captures",
    "outcome",
    "attestation_digest",
})
_CAPTURE_KEYS = frozenset({
    "schema",
    "modality",
    "normalized_command_digest",
    "fixture_prompt_digest",
    "fixture_schema_digest",
    "fixture_image_digest",
    "request_method",
    "request_path",
    "request_content_encoding",
    "request_body_raw_base64",
    "request_body_raw_digest",
    "request_body_canonical_digest",
    "model",
    "responses_lite",
    "top_level_tools_absent",
    "additional_tools_count",
    "tool_surface_digest",
    "tool_choice",
    "parallel_tool_calls",
    "outcome",
})


@dataclass(frozen=True)
class CodexNoToolsAttestation:
    """Deep-immutable canonical bytes suitable for a campaign precommit."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes) \
                or not 0 < len(self.canonical_bytes) <= _MAX_ATTESTATION_BYTES:
            raise T.CodexProposerFailure(
                "Codex no-tools attestation must be bounded canonical bytes")
        value = _decode_attestation(self.canonical_bytes)
        if T._canonical_json_bytes(value) != self.canonical_bytes:
            raise T.CodexProposerFailure(
                "Codex no-tools attestation bytes are not canonical")
        _validate_attestation_record(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodexNoToolsAttestation":
        if not isinstance(value, Mapping):
            raise T.CodexProposerFailure(
                "Codex no-tools attestation must be a mapping")
        return cls(T._canonical_json_bytes(dict(value)))

    @property
    def attestation_digest(self) -> str:
        return self.to_dict()["attestation_digest"]

    @property
    def launcher_digest(self) -> str:
        return self.to_dict()["launcher_digest"]

    @property
    def model_catalog_digest(self) -> str:
        return self.to_dict()["model_catalog_digest"]

    @property
    def cloud_config_bundle_cache_binding(self) -> str:
        return self.to_dict()["cloud_config_bundle_cache_binding"]

    def to_dict(self) -> dict[str, Any]:
        return _decode_attestation(self.canonical_bytes)


def _decode_attestation(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise T.CodexProposerFailure(
            "Codex no-tools attestation is not UTF-8") from exc
    value = T._strict_json(text, "no-tools attestation")
    if not isinstance(value, dict):
        raise T.CodexProposerFailure(
            "Codex no-tools attestation must be an object")
    return value


def _validate_hash(value: Any, description: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise T.CodexProposerFailure(
            f"Codex no-tools {description} is not SHA-256")


def _validate_request_modality(request: Mapping[str, Any], modality: str) -> None:
    inputs = request.get("input")
    if not isinstance(inputs, list):
        raise T.CodexProposerFailure(
            "Codex no-tools request input is not a list")
    content = [
        item
        for message in inputs
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict)
    ]
    prompt_items = [
        item for item in content
        if item == {"type": "input_text", "text": _CAPTURE_PROMPT}
    ]
    image_items = [item for item in content if item.get("type") == "input_image"]
    expected_text = {
        "format": {
            "name": "codex_output_schema",
            "schema": _CAPTURE_SCHEMA,
            "strict": True,
            "type": "json_schema",
        },
        "verbosity": "low",
    }
    if len(prompt_items) != 1 or request.get("text") != expected_text:
        raise T.CodexProposerFailure(
            "Codex no-tools request does not bind the synthetic prompt/schema")
    if modality == "text":
        if image_items:
            raise T.CodexProposerFailure(
                "Codex text preflight request unexpectedly contains an image")
        return
    expected_image = {
        "type": "input_image",
        "image_url": (
            "data:image/png;base64,"
            + base64.b64encode(_CAPTURE_PNG).decode("ascii")),
    }
    wrappers = [
        item.get("text") for item in content
        if item.get("type") == "input_text"
        and isinstance(item.get("text"), str)
        and item["text"].startswith('<image name=[Image #1] path="')
    ]
    if image_items != [expected_image] or len(wrappers) != 1 \
            or not wrappers[0].endswith('/synthetic.png">') \
            or sum(item == {"type": "input_text", "text": "</image>"}
                   for item in content) != 1:
        raise T.CodexProposerFailure(
            "Codex image preflight request does not bind the synthetic PNG")


def _validate_capture_row(row: Any, modality: str) -> None:
    if not isinstance(row, dict) or set(row) != _CAPTURE_KEYS:
        raise T.CodexProposerFailure(
            "Codex no-tools capture row fields are invalid")
    expected_image_digest = (
        "" if modality == "text" else T._bytes_digest(_CAPTURE_PNG))
    if row["schema"] != _CAPTURE_ROW_SCHEMA \
            or row["modality"] != modality \
            or row["fixture_prompt_digest"] != \
            T._bytes_digest(_CAPTURE_PROMPT.encode("utf-8")) \
            or row["fixture_schema_digest"] != \
            T._bytes_digest(T._canonical_json_bytes(_CAPTURE_SCHEMA)) \
            or row["fixture_image_digest"] != expected_image_digest \
            or row["normalized_command_digest"] != \
            T._codex_command_digest(_normalized_capture_command(modality)) \
            or row["request_method"] != "POST" \
            or row["request_path"] != "/v1/responses" \
            or row["request_content_encoding"] != "absent" \
            or row["model"] != T.DEFAULT_CODEX_MODEL \
            or row["responses_lite"] is not True \
            or row["top_level_tools_absent"] is not True \
            or row["additional_tools_count"] != 1 \
            or row["tool_surface_digest"] != T.CODEX_TOOL_SURFACE_DIGEST \
            or row["tool_choice"] != "auto" \
            or row["parallel_tool_calls"] is not False \
            or row["outcome"] != "success":
        raise T.CodexProposerFailure(
            "Codex no-tools capture row policy differs")
    for key in (
        "normalized_command_digest",
        "fixture_prompt_digest",
        "fixture_schema_digest",
        "request_body_raw_digest",
        "request_body_canonical_digest",
        "tool_surface_digest",
    ):
        _validate_hash(row[key], key)
    encoded_body = row["request_body_raw_base64"]
    if not isinstance(encoded_body, str):
        raise T.CodexProposerFailure(
            "Codex no-tools captured request body is not base64 text")
    try:
        raw_body = base64.b64decode(encoded_body, validate=True)
    except (ValueError, TypeError) as exc:
        raise T.CodexProposerFailure(
            "Codex no-tools captured request body base64 is malformed") from exc
    if not raw_body or len(raw_body) > _MAX_CAPTURE_REQUEST_BYTES \
            or T._bytes_digest(raw_body) != row["request_body_raw_digest"]:
        raise T.CodexProposerFailure(
            "Codex no-tools captured request raw digest differs")
    try:
        request = T._strict_json(
            raw_body.decode("utf-8", errors="strict"),
            "attested Responses request",
        )
    except UnicodeError as exc:
        raise T.CodexProposerFailure(
            "Codex no-tools attested request is not UTF-8") from exc
    if not isinstance(request, dict) \
            or T._digest(request) != row["request_body_canonical_digest"] \
            or "tools" in request \
            or request.get("model") != T.DEFAULT_CODEX_MODEL \
            or request.get("tool_choice") != "auto" \
            or request.get("parallel_tool_calls") is not False:
        raise T.CodexProposerFailure(
            "Codex no-tools attested request policy differs")
    _validate_request_modality(request, modality)
    inputs = request.get("input")
    additional = [item for item in inputs or []
                  if isinstance(item, dict)
                  and item.get("type") == "additional_tools"] \
        if isinstance(inputs, list) else []
    expected_additional = {
        "type": "additional_tools",
        "role": "developer",
        "tools": [],
    }
    if not isinstance(inputs, list) or len(additional) != 1 \
            or not inputs or inputs[0] != expected_additional \
            or additional[0] != expected_additional \
            or T._digest(additional[0]["tools"]) != \
            row["tool_surface_digest"]:
        raise T.CodexProposerFailure(
            "Codex no-tools attested additional-tools surface differs")


def _validate_attestation_record(value: Mapping[str, Any]) -> None:
    if set(value) != _ATTESTATION_KEYS:
        raise T.CodexProposerFailure(
            "Codex no-tools attestation fields are invalid")
    if value["schema"] != CODEX_NO_TOOLS_ATTESTATION_SCHEMA \
            or value["launcher_version"] != T.PINNED_CODEX_CLI_VERSION \
            or value["capture_harness_source_digest"] != \
            _LOADED_SOURCE_SHA256 \
            or value["model_catalog_digest"] != \
            T.PINNED_MODEL_CATALOG_RAW_DIGEST \
            or value["model_catalog_canonical_digest"] != \
            T.PINNED_MODEL_CATALOG_CANONICAL_DIGEST \
            or value["bundled_catalog_raw_digest"] != \
            T.PINNED_BUNDLED_CATALOG_RAW_DIGEST \
            or value["bundled_model_record_canonical_digest"] != \
            T.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST \
            or value["catalog_diff_policy"] != \
            "only-tool_mode-direct-and-apply_patch_tool_type-null" \
            or value["transport_policy_digest"] != \
            T.CODEX_TRANSPORT_POLICY_DIGEST \
            or value["effective_tool_mode"] != T.CODEX_EFFECTIVE_TOOL_MODE \
            or value["apply_patch_tool_type"] != \
            T.CODEX_APPLY_PATCH_TOOL_TYPE \
            or value["tool_surface_digest"] != \
            T.CODEX_TOOL_SURFACE_DIGEST \
            or value["modality_coverage"] != ["text", "named_image"] \
            or value["outcome"] != "success":
        raise T.CodexProposerFailure(
            "Codex no-tools attestation policy identity differs")
    for key in (
        "launcher_digest",
        "capture_harness_source_digest",
        "model_catalog_digest",
        "model_catalog_canonical_digest",
        "bundled_catalog_raw_digest",
        "bundled_model_record_canonical_digest",
        "transport_policy_digest",
        "tool_surface_digest",
        "attestation_digest",
    ):
        _validate_hash(value[key], key)
    policy_binding = value["cloud_config_bundle_cache_binding"]
    if policy_binding != "absent" and (
        not isinstance(policy_binding, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", policy_binding) is None
    ):
        raise T.CodexProposerFailure(
            "Codex no-tools cloud policy binding is invalid")
    captures = value["captures"]
    if not isinstance(captures, list) or len(captures) != 2:
        raise T.CodexProposerFailure(
            "Codex no-tools attestation must contain two captures")
    for row, modality in zip(captures, ("text", "named_image")):
        _validate_capture_row(row, modality)
    body = {key: item for key, item in value.items()
            if key != "attestation_digest"}
    if value["attestation_digest"] != T._digest(body):
        raise T.CodexProposerFailure(
            "Codex no-tools attestation digest does not reproduce")


def validate_codex_no_tools_attestation(
        attestation: CodexNoToolsAttestation | Mapping[str, Any], *,
        expected_launcher_digest: str | None = None,
        expected_model_catalog_digest: str | None = None,
        expected_cloud_policy_cache_binding: str | None = None,
        ) -> CodexNoToolsAttestation:
    frozen = (
        attestation if isinstance(attestation, CodexNoToolsAttestation)
        else CodexNoToolsAttestation.from_mapping(attestation))
    record = frozen.to_dict()
    expected = {
        "launcher_digest": expected_launcher_digest,
        "model_catalog_digest": expected_model_catalog_digest,
        "cloud_config_bundle_cache_binding": (
            expected_cloud_policy_cache_binding),
    }
    for key, wanted in expected.items():
        if wanted is not None and record[key] != wanted:
            raise T.CodexProposerFailure(
                f"Codex no-tools attestation {key} differs")
    return frozen


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CaptureHandler)
        self.captures: list[dict[str, Any]] = []


class _CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _record_and_reject(self) -> None:
        server = self.server
        assert isinstance(server, _CaptureServer)
        capture: dict[str, Any] = {
            "method": self.command,
            "path": self.path,
            "content_encoding": self.headers.get_all("Content-Encoding") or [],
        }
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isdigit():
            capture["error"] = "missing or duplicate Content-Length"
        else:
            length = int(lengths[0])
            if not 0 < length <= _MAX_CAPTURE_REQUEST_BYTES:
                capture["error"] = "request body is empty or oversized"
            else:
                capture["body"] = self.rfile.read(length)
                if len(capture["body"]) != length:
                    capture["error"] = "request body was truncated"
        server.captures.append(capture)
        response = T._canonical_json_bytes({
            "error": {
                "type": "bongard_local_capture_complete",
                "message": "local no-tools request captured",
            }
        })
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response)
        self.close_connection = True

    do_POST = _record_and_reject
    do_GET = _record_and_reject
    do_PUT = _record_and_reject
    do_DELETE = _record_and_reject


def _write_exact(path: str, data: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise T.CodexProposerFailure(
                    "could not completely stage Codex preflight fixture")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _capture_provider_overrides(base_url: str) -> tuple[str, ...]:
    prefix = f"model_providers.{_CAPTURE_PROVIDER_ID}"
    return (
        f'model_provider="{_CAPTURE_PROVIDER_ID}"',
        f'{prefix}.name="Bongard local no-tools capture"',
        f"{prefix}.base_url=" + json.dumps(base_url),
        f'{prefix}.wire_api="responses"',
        f"{prefix}.requires_openai_auth=false",
        f"{prefix}.supports_websockets=false",
        f"{prefix}.request_max_retries=0",
        f"{prefix}.stream_max_retries=0",
        "analytics.enabled=false",
    )


def _normalized_capture_command(modality: str) -> tuple[str, ...]:
    image_paths: Sequence[str] = () if modality == "text" else ("<IMAGE>",)
    return T._codex_command(
        executable="<CODEX-0.147.0>",
        view_dir="<PRIVATE-VIEW>",
        image_paths=image_paths,
        schema_path="<OUTPUT-SCHEMA>",
        model_catalog_path="<MODEL-CATALOG>",
        model=T.DEFAULT_CODEX_MODEL,
        reasoning_effort=T.DEFAULT_REASONING_EFFORT,
        extra_config_overrides=_capture_provider_overrides(
            "http://127.0.0.1:<PORT>/v1"),
    )


def _require_exactly_one_capture(
        captures: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(captures) != 1:
        raise T.CodexProposerFailure(
            "Codex capture did not issue exactly one local request")
    return captures[0]


def _capture_row(
        *, modality: str, executable: str,
        model_catalog_snapshot: T.CodexModelCatalogSnapshot,
        cloud_policy_cache_snapshot: T.CloudPolicyCacheSnapshot,
        temp_parent: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
            prefix=f"bongard-codex-capture-{modality}-auth-",
            dir=temp_parent) as auth_dir, tempfile.TemporaryDirectory(
            prefix=f"bongard-codex-capture-{modality}-view-",
            dir=temp_parent) as view_dir:
        T._require_outside_bongard(auth_dir, "Codex capture auth home")
        T._require_outside_bongard(view_dir, "Codex capture private view")
        os.chmod(auth_dir, 0o700)
        os.chmod(view_dir, 0o700)
        policy_cache = T._stage_cloud_policy_cache(
            auth_dir, cloud_policy_cache_snapshot)
        model_catalog = T._stage_model_catalog(
            view_dir, model_catalog_snapshot)
        schema_bytes = T._canonical_json_bytes(_CAPTURE_SCHEMA)
        schema_path = os.path.join(view_dir, "output_schema.json")
        _write_exact(schema_path, schema_bytes)
        image_paths: tuple[str, ...] = ()
        if modality == "named_image":
            image_path = os.path.join(view_dir, "synthetic.png")
            _write_exact(image_path, _CAPTURE_PNG)
            image_paths = (image_path,)

        server = _CaptureServer()
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        port = int(server.server_address[1])
        base_url = f"http://127.0.0.1:{port}/v1"
        command = T._codex_command(
            executable=executable,
            view_dir=view_dir,
            image_paths=image_paths,
            schema_path=schema_path,
            model_catalog_path=model_catalog.path,
            model=T.DEFAULT_CODEX_MODEL,
            reasoning_effort=T.DEFAULT_REASONING_EFFORT,
            extra_config_overrides=_capture_provider_overrides(base_url),
        )
        environment = T._minimal_environment(
            codex_home=auth_dir, temp_parent=temp_parent)
        for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "CODEX_API_KEY",
        ):
            environment.pop(key, None)
        environment["NO_PROXY"] = "127.0.0.1,localhost"
        environment["no_proxy"] = "127.0.0.1,localhost"
        try:
            T._recheck_staged_cloud_policy_cache(policy_cache)
            T._recheck_staged_model_catalog(model_catalog)
            returncode, _stdout, _stderr = T._run_codex_process(
                command,
                task_bytes=_CAPTURE_PROMPT.encode("utf-8"),
                view_dir=view_dir,
                environment=environment,
                minutes=1,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        T._recheck_staged_cloud_policy_cache(policy_cache)
        T._recheck_staged_model_catalog(model_catalog)
        if returncode == 0:
            raise T.CodexProposerFailure(
                "Codex capture unexpectedly accepted the mock error response")
        if thread.is_alive():
            raise T.CodexProposerFailure(
                "Codex capture server did not terminate")
        capture = _require_exactly_one_capture(server.captures)
        if capture.get("error") is not None \
                or capture.get("method") != "POST" \
                or capture.get("path") != "/v1/responses" \
                or capture.get("content_encoding") != []:
            raise T.CodexProposerFailure(
                "Codex capture request framing differs from policy")
        raw_body = capture.get("body")
        if not isinstance(raw_body, bytes):
            raise T.CodexProposerFailure("Codex capture body is missing")
        try:
            body_text = raw_body.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise T.CodexProposerFailure(
                "Codex capture body is not UTF-8") from exc
        request = T._strict_json(body_text, "captured Responses request")
        if not isinstance(request, dict):
            raise T.CodexProposerFailure(
                "Codex captured Responses request is not an object")
        if "tools" in request \
                or request.get("model") != T.DEFAULT_CODEX_MODEL \
                or request.get("tool_choice") != "auto" \
                or request.get("parallel_tool_calls") is not False:
            raise T.CodexProposerFailure(
                "Codex captured Responses tool policy differs")
        inputs = request.get("input")
        additional = [item for item in inputs or []
                      if isinstance(item, dict)
                      and item.get("type") == "additional_tools"] \
            if isinstance(inputs, list) else []
        expected_additional = {
            "type": "additional_tools",
            "role": "developer",
            "tools": [],
        }
        if not isinstance(inputs, list) or len(additional) != 1 \
                or not inputs or inputs[0] != expected_additional \
                or additional[0] != expected_additional:
            raise T.CodexProposerFailure(
                "Codex Responses Lite additional-tools surface is not empty")
        _validate_request_modality(request, modality)
        row: dict[str, Any] = {
            "schema": _CAPTURE_ROW_SCHEMA,
            "modality": modality,
            "normalized_command_digest": T._codex_command_digest(
                _normalized_capture_command(modality)),
            "fixture_prompt_digest": T._bytes_digest(
                _CAPTURE_PROMPT.encode("utf-8")),
            "fixture_schema_digest": T._bytes_digest(schema_bytes),
            "fixture_image_digest": (
                "" if modality == "text" else T._bytes_digest(_CAPTURE_PNG)),
            "request_method": "POST",
            "request_path": "/v1/responses",
            "request_content_encoding": "absent",
            "request_body_raw_base64": base64.b64encode(raw_body).decode("ascii"),
            "request_body_raw_digest": T._bytes_digest(raw_body),
            "request_body_canonical_digest": T._digest(request),
            "model": request["model"],
            "responses_lite": True,
            "top_level_tools_absent": True,
            "additional_tools_count": 1,
            "tool_surface_digest": T._digest(additional[0]["tools"]),
            "tool_choice": request["tool_choice"],
            "parallel_tool_calls": request["parallel_tool_calls"],
            "outcome": "success",
        }
        _validate_capture_row(row, modality)
        return row


def _verify_bundled_catalog(
        executable: str, *, temp_parent: str) -> None:
    with tempfile.TemporaryDirectory(
            prefix="bongard-codex-bundled-catalog-", dir=temp_parent
            ) as codex_home, tempfile.TemporaryFile(
                dir=temp_parent) as stdout_file, tempfile.TemporaryFile(
                dir=temp_parent) as stderr_file:
        environment = T._minimal_environment(
            codex_home=codex_home, temp_parent=temp_parent)
        process = subprocess.run(
            [executable, "debug", "models", "--bundled"],
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=temp_parent,
            env=environment,
            check=False,
            timeout=30,
        )
        if process.returncode != 0:
            raise T.CodexProposerFailure(
                "cannot inspect pinned Codex bundled model catalog")
        stdout_file.seek(0)
        raw = stdout_file.read(1_000_001)
    if len(raw) > 1_000_000 \
            or T._bytes_digest(raw) != T.PINNED_BUNDLED_CATALOG_RAW_DIGEST:
        raise T.CodexProposerFailure(
            "pinned Codex bundled model catalog bytes differ")
    try:
        decoded = T._strict_json(
            raw.decode("utf-8", errors="strict"), "bundled model catalog")
    except UnicodeError as exc:
        raise T.CodexProposerFailure(
            "pinned Codex bundled model catalog is not UTF-8") from exc
    models = decoded.get("models") if isinstance(decoded, dict) else None
    matches = [item for item in models or []
               if isinstance(item, dict)
               and item.get("slug") == T.DEFAULT_CODEX_MODEL] \
        if isinstance(models, list) else []
    if len(matches) != 1 or T._digest(matches[0]) != \
            T.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST:
        raise T.CodexProposerFailure(
            "pinned Codex bundled Sol record differs")


def attest_codex_no_tools(
        *, executable: str, expected_launcher_digest: str,
        model_catalog_snapshot: T.CodexModelCatalogSnapshot,
        cloud_policy_cache_snapshot: T.CloudPolicyCacheSnapshot,
        ) -> CodexNoToolsAttestation:
    """Capture and freeze both live request modalities before corpus release."""

    if not isinstance(model_catalog_snapshot, T.CodexModelCatalogSnapshot):
        raise T.CodexProposerFailure("Codex model catalog snapshot type is invalid")
    if not isinstance(cloud_policy_cache_snapshot, T.CloudPolicyCacheSnapshot):
        raise T.CodexProposerFailure("Codex cloud policy snapshot type is invalid")
    temp_parent = T._safe_temp_parent()
    with T.stage_codex_launcher(
            executable,
            expected_launcher_digest=expected_launcher_digest) as staged:
        T._require_pinned_cli_version(staged.version)
        _verify_bundled_catalog(staged.executable, temp_parent=temp_parent)
        captures = [
            _capture_row(
                modality=modality,
                executable=staged.executable,
                model_catalog_snapshot=model_catalog_snapshot,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                temp_parent=temp_parent,
            )
            for modality in ("text", "named_image")
        ]
        body: dict[str, Any] = {
            "schema": CODEX_NO_TOOLS_ATTESTATION_SCHEMA,
            "launcher_version": staged.version,
            "launcher_digest": staged.launcher_digest,
            "capture_harness_source_digest": _LOADED_SOURCE_SHA256,
            "model_catalog_digest": model_catalog_snapshot.raw_digest,
            "model_catalog_canonical_digest": (
                model_catalog_snapshot.canonical_digest),
            "bundled_catalog_raw_digest": T.PINNED_BUNDLED_CATALOG_RAW_DIGEST,
            "bundled_model_record_canonical_digest": (
                T.PINNED_BUNDLED_MODEL_RECORD_CANONICAL_DIGEST),
            "catalog_diff_policy": (
                "only-tool_mode-direct-and-apply_patch_tool_type-null"),
            "cloud_config_bundle_cache_binding": (
                cloud_policy_cache_snapshot.binding),
            "transport_policy_digest": T.CODEX_TRANSPORT_POLICY_DIGEST,
            "effective_tool_mode": T.CODEX_EFFECTIVE_TOOL_MODE,
            "apply_patch_tool_type": T.CODEX_APPLY_PATCH_TOOL_TYPE,
            "tool_surface_digest": T.CODEX_TOOL_SURFACE_DIGEST,
            "modality_coverage": ["text", "named_image"],
            "captures": captures,
            "outcome": "success",
        }
        body["attestation_digest"] = T._digest(body)
        return CodexNoToolsAttestation.from_mapping(body)


__all__ = (
    "CODEX_NO_TOOLS_ATTESTATION_SCHEMA",
    "CodexNoToolsAttestation",
    "attest_codex_no_tools",
    "validate_codex_no_tools_attestation",
)
