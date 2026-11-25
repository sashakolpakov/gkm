from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc.crack_lab import arc_agi3_codex_app_server_transport as T
from arc.crack_lab import arc_agi3_contiguous_taint as Taint


def _auth_document() -> dict[str, object]:
    return {
        "OPENAI_API_KEY": "unused-api-key-sentinel",
        "auth_mode": "chatgpt",
        "last_refresh": "2026-07-28T00:00:00Z",
        "tokens": {
            "access_token": "access-token-sentinel",
            "account_id": "account-id-sentinel",
            "id_token": "id-token-sentinel",
            "refresh_token": "refresh-token-sentinel",
        },
    }


def _write_private_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(T.AppServerTransportError, match="duplicate"):
        T.strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(T.AppServerTransportError, match="non-finite"):
        T.strict_json_loads('{"a":NaN}')
    assert T.strict_json_loads('{"a":1}') == {"a": 1}


def test_external_credentials_are_descriptor_read_and_redacted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "auth.json"
    document = _auth_document()
    _write_private_json(source, document)

    credentials = T.load_external_chatgpt_credentials(source)

    assert credentials.login_params() == {
        "type": "chatgptAuthTokens",
        "accessToken": "access-token-sentinel",
        "chatgptAccountId": "account-id-sentinel",
    }
    redacted = T.canonical_json(credentials.redacted_login_params())
    for sentinel in credentials.leak_sentinels:
        assert sentinel.encode("utf-8") not in redacted
    assert credentials.leak_sentinels == (
        "unused-api-key-sentinel",
        "access-token-sentinel",
        "account-id-sentinel",
        "id-token-sentinel",
        "refresh-token-sentinel",
    )
    assert len(credentials.redacted_request_sha256()) == 64


def test_external_credentials_reject_permissions_symlinks_and_schema_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "auth.json"
    _write_private_json(source, _auth_document())
    source.chmod(0o640)
    with pytest.raises(T.AppServerTransportError, match="inadmissible"):
        T.load_external_chatgpt_credentials(source)

    source.chmod(0o600)
    alias = tmp_path / "auth-alias.json"
    alias.symlink_to(source)
    with pytest.raises(T.AppServerTransportError):
        T.load_external_chatgpt_credentials(alias)

    drifted = _auth_document()
    drifted["unexpected"] = True
    _write_private_json(source, drifted)
    with pytest.raises(T.AppServerTransportError, match="pinned"):
        T.load_external_chatgpt_credentials(source)


def _refresh_test_controller(
    credentials: T.ExternalChatGptCredentials,
    transcript_path: Path,
) -> T.CodexAppServerController:
    controller = object.__new__(T.CodexAppServerController)
    state_root = transcript_path.parent / "controller-state"
    state_root.mkdir(exist_ok=True)
    controller.binding = SimpleNamespace(
        max_auth_refreshes=7,
        state_root=str(state_root),
    )
    controller.credentials = credentials
    controller._refresh_count = 0
    controller._credential_sentinels = set(
        credentials.leak_sentinels
    )
    controller._credential_access_token_sha256 = {
        T.sha256_bytes(credentials.access_token.encode("utf-8"))
    }
    controller._stderr_complete = bytearray()
    controller._refresh_redacted_response_sha256 = []
    controller.transcript = SimpleNamespace(path=transcript_path)
    controller._write_wire = lambda *_args, **_kwargs: None
    return controller


def test_auth_refresh_requires_a_genuinely_new_access_token(
    tmp_path: Path,
) -> None:
    source = tmp_path / "auth.json"
    document = _auth_document()
    _write_private_json(source, document)
    credentials = T.load_external_chatgpt_credentials(source)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(b"{}\n")
    controller = _refresh_test_controller(
        credentials, transcript
    )
    event = {
        "id": "refresh-1",
        "params": {
            "reason": "unauthorized",
            "previousAccountId": credentials.account_id,
        },
    }
    with pytest.raises(
        T.AppServerTransportError,
        match="did not rotate",
    ):
        controller._handle_auth_refresh(event)
    assert controller._refresh_count == 0

    for index in range(1, 4):
        rotated = _auth_document()
        rotated["tokens"]["access_token"] = f"rotated-token-{index}"
        _write_private_json(source, rotated)
        controller._handle_auth_refresh(
            {
                "id": f"refresh-{index + 1}",
                "params": {
                    "reason": "unauthorized",
                    "previousAccountId": credentials.account_id,
                },
            }
        )
    assert controller._refresh_count == 3
    assert len(controller._credential_access_token_sha256) == 4
    assert len(controller._refresh_redacted_response_sha256) == 3
    assert {
        "rotated-token-1",
        "rotated-token-2",
        "rotated-token-3",
    }.issubset(controller.credential_sentinels_for_host_scan())


def test_os_process_start_identity_is_stable_and_birth_bound() -> None:
    first = T.observe_os_process_start_identity(os.getpid())
    second = T.observe_os_process_start_identity(os.getpid())
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("solver.py", True),
        ("nested/solver.py", True),
        ("a/../x", False),
        ("a/.", False),
        ("a/..", False),
        ("../x", False),
        ("/absolute", False),
        ("a//b", False),
        ("a\\b", False),
        (".hidden", False),
        ("a/.hidden", False),
        ("a\x00b", False),
        ("", False),
        (None, False),
    ),
)
def test_relative_path_admission_is_closed(
    value: object, expected: bool
) -> None:
    assert T.is_safe_relative_path(value) is expected


def test_dynamic_tool_schemas_are_closed_and_unsafe_exec_is_not_admitted() -> None:
    names = T.DYNAMIC_TOOL_NAMES
    assert names == (
        *(row[0] for row in T._TOOL_ROWS),
        "workspace_run_python",
    )
    assert len(names) == len(set(names))
    assert T.BRIDGE_EXEC_ALLOWLIST == ()
    assert len(T.DYNAMIC_TOOL_SPECS) == 1
    namespace = T.DYNAMIC_TOOL_SPECS[0]
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "contiguous_lane"
    assert tuple(tool["name"] for tool in namespace["tools"]) == names
    assert all(
        row["type"] == "function"
        and row["deferLoading"] is False
        and row["inputSchema"]["additionalProperties"] is False
        for row in namespace["tools"]
    )
    assert T.DYNAMIC_TOOL_SPECS_SHA256 == hashlib.sha256(
        T.canonical_json(T.DYNAMIC_TOOL_SPECS)
    ).hexdigest()

    probe = T.SAFE_PROBE_TOOL_SPEC
    assert probe["name"] == "workspace_run_python"
    assert probe["inputSchema"]["additionalProperties"] is False
    assert set(probe["inputSchema"]["required"]) == {
        "entrypoint",
        "files",
        "arguments",
        "timeout_seconds",
    }


def test_probe_stderr_traceback_is_host_only_and_visibility_receipt_is_exact(
    tmp_path: Path,
) -> None:
    raw = (
        b"Traceback (most recent call last):\n"
        b'  File "/private/harness/gkm_arena.py", line 411, in step\n'
        b"    return private_engine_transition()\n"
        b"KeyboardInterrupt\n"
    )
    stderr_path = tmp_path / "probe" / "evidence" / "stderr.bin"
    stderr_path.parent.mkdir(parents=True)
    stderr_path.write_bytes(raw)
    stderr_path.chmod(0o400)

    visible, body = T._probe_stderr_visibility_projection(raw)
    assert visible == T.PROBE_STDERR_SANITIZED_LINE
    assert "gkm_arena" not in visible
    assert "private_engine_transition" not in visible
    visible_path = tmp_path / "proposer-visible-stderr.txt"
    visible_path.write_text(visible, encoding="utf-8")
    assert Taint.scan_evidence(
        visible_path, evidence_kind="container_stderr"
    ).hits == ()
    assert body == {
        "schema": 1,
        "kind": T.PROBE_STDERR_VISIBILITY_KIND,
        "raw_stderr_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_stderr_bytes": len(raw),
        "raw_surface_classification":
            "python_or_harness_traceback",
        "raw_bytes_host_only": True,
        "proposer_visible_stderr": T.PROBE_STDERR_SANITIZED_LINE,
        "proposer_visible_stderr_sha256": hashlib.sha256(
            T.PROBE_STDERR_SANITIZED_LINE.encode("utf-8")
        ).hexdigest(),
        "proposer_visible_traceback_absent": True,
        "proposer_visible_taint_status": "CLEAN",
    }
    receipt_path, receipt_sha256 = (
        T._retain_probe_stderr_visibility_receipt(stderr_path, body)
    )
    receipt = Path(receipt_path)
    receipt_raw = receipt.read_bytes()
    assert receipt.stat().st_mode & 0o777 == 0o400
    assert hashlib.sha256(receipt_raw).hexdigest() == receipt_sha256
    assert hashlib.sha256(stderr_path.read_bytes()).hexdigest() == (
        body["raw_stderr_sha256"]
    )
    # Reopening is idempotent; mutation is rejected rather than laundered.
    assert T._retain_probe_stderr_visibility_receipt(
        stderr_path, body
    ) == (receipt_path, receipt_sha256)
    with pytest.raises(
        T.AppServerTransportError,
        match="visibility receipt differs",
    ):
        T._retain_probe_stderr_visibility_receipt(
            stderr_path, {**body, "raw_stderr_bytes": len(raw) + 1}
        )


def test_snapshot_manifest_parser_binds_exact_generation_call_and_entries(
    tmp_path: Path,
) -> None:
    generation = (tmp_path / "generation").resolve()
    call = generation / "probe_calls" / "request-007" / "call-001"
    snapshot = call / "snapshot"
    snapshot.mkdir(parents=True)
    entry_path = snapshot / "solver.py"
    entry_path.write_text("print('ok')\n", encoding="utf-8")
    snapshot_stat = snapshot.stat()
    entry_stat = entry_path.stat()
    entry_bytes = entry_path.read_bytes()
    value = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_workspace_snapshot",
        "campaign_id": "campaign",
        "generation_id": "generation",
        "attempt_id": "attempt",
        "dynamic_request_id": 7,
        "dynamic_call_id": "call-001",
        "thread_id": "thread",
        "turn_id": "turn",
        "generation_dir": str(generation),
        "call_dir": str(call),
        "snapshot_root": str(snapshot),
        "snapshot_device": snapshot_stat.st_dev,
        "snapshot_inode": snapshot_stat.st_ino,
        "tree_sha256": "1" * 64,
        "entries": [
            {
                "path": "solver.py",
                "sha256": hashlib.sha256(entry_bytes).hexdigest(),
                "bytes": len(entry_bytes),
                "device": entry_stat.st_dev,
                "inode": entry_stat.st_ino,
            }
        ],
        "source_workspace_tree_sha256": "2" * 64,
        "no_writeback": True,
    }

    parsed = T.workspace_snapshot_manifest_from_dict(value)
    assert parsed.call_dir == str(call)
    assert parsed.snapshot_root == str(snapshot)
    assert asdict(parsed)["entries"][0]["path"] == "solver.py"

    escaped = json.loads(json.dumps(value))
    escaped["entries"][0]["path"] = "../solver.py"
    with pytest.raises(T.AppServerTransportError, match="entry"):
        T.workspace_snapshot_manifest_from_dict(escaped)

    wrong_call = json.loads(json.dumps(value))
    wrong_call["call_dir"] = str(generation / "call-001")
    with pytest.raises(T.AppServerTransportError, match="schema"):
        T.workspace_snapshot_manifest_from_dict(wrong_call)


def test_chained_transcript_is_exclusive_and_digest_linked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "app_server.jsonl"
    transcript = T.ChainedTranscript(path)
    first = transcript.append(
        direction="client_request",
        payload={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )
    second = transcript.append(
        direction="server_response",
        payload={"jsonrpc": "2.0", "id": 1, "result": {}},
    )
    transcript.close()

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["sequence"] == 1
    assert rows[0]["previous_digest"] is None
    assert rows[0]["digest"] == first
    assert rows[1]["sequence"] == 2
    assert rows[1]["previous_digest"] == first
    assert rows[1]["digest"] == second
    for row in rows:
        digest = row.pop("digest")
        assert digest == hashlib.sha256(T.canonical_json(row)).hexdigest()
    with pytest.raises(FileExistsError):
        T.ChainedTranscript(path)


def test_config_projection_is_zero_ambient_and_fail_closed() -> None:
    rendered = T.render_strict_config(
        model="gpt-5.6-sol",
        model_provider="openai",
        effort="max",
    )
    assert 'approval_policy = "never"' in rendered
    assert 'sandbox_mode = "read-only"' in rendered
    assert 'web_search = "disabled"' in rendered
    assert 'inherit = "none"' in rendered
    assert "enabled = true" not in rendered
    assert "mcp_servers" not in rendered
    with pytest.raises(T.AppServerTransportError):
        T.render_strict_config(
            model="gpt-5.6-sol",
            model_provider="other",
            effort="max",
        )
    with pytest.raises(T.AppServerTransportError):
        T.render_strict_config(
            model="gpt-5.6-sol",
            model_provider="openai",
            effort="ultra",
        )
