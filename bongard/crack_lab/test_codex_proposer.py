"""Offline adversarial tests for the isolated headless Codex transport.

No test in this module launches Codex or makes a model/API request.  Both the
version probe and the non-interactive turn are replaced at the subprocess
boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codex_proposer as C
import semantic_replay as SR


THREAD_ID = "12345678-1234-4abc-9234-abcdef123456"
MODEL = "gpt-5.6-sol"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.fixture
def panel_paths(tmp_path):
    paths = []
    for prefix in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{prefix}_{index}.png"
            presentation = np.full((16, 16), 255, dtype=np.uint8)
            presentation[index, 0 if prefix == "pos" else 1] = 0
            Image.fromarray(presentation, mode="L").save(path, format="PNG")
            paths.append(str(path.resolve()))
    return paths


def _jsonl(
        payload: Any = None,
        *,
        reported_model: str | None = None,
        usage: dict[str, Any] | None = None,
        item_type: str = "agent_message",
        messages: int = 1,
        extra_events: list[dict[str, Any]] | None = None,
        thread_id: str = THREAD_ID) -> bytes:
    if payload is None:
        payload = {"answer": "ok"}
    started: dict[str, Any] = {
        "type": "thread.started", "thread_id": thread_id,
    }
    if reported_model is not None:
        started["model"] = reported_model
    events: list[dict[str, Any]] = [started, {"type": "turn.started"}]
    events.extend(extra_events or [])
    for index in range(messages):
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        events.append({
            "type": "item.completed",
            "item": {
                "id": f"item_{index}", "type": item_type, "text": text,
            },
        })
    events.append({
        "type": "turn.completed",
        "usage": usage or {
            "input_tokens": 17,
            "cached_input_tokens": 3,
            "output_tokens": 5,
            "reasoning_output_tokens": 2,
        },
    })
    return ("\n".join(json.dumps(event, separators=(",", ":"))
                       for event in events) + "\n").encode()


def _write_mock_stream(kwargs: dict[str, Any], name: str, value: bytes) \
        -> bytes | None:
    """Support both capture_output and bounded-tempfile implementations."""
    stream = kwargs.get(name)
    if stream is not None and hasattr(stream, "write"):
        stream.write(value)
        stream.flush()
        return None
    return value


def _install_cli(
        monkeypatch,
        stdout: bytes,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        inspect_exec=None,
        after_exec=None,
        exec_exception: BaseException | None = None):
    class CallLog(list):
        processes: list[Any]

    calls = CallLog()
    calls.processes = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if len(command) == 2 and command[1] == "--version":
            version_stdout = _write_mock_stream(
                kwargs, "stdout", b"codex-cli 0.146.0\n")
            version_stderr = _write_mock_stream(kwargs, "stderr", b"")
            return subprocess.CompletedProcess(
                command, 0, stdout=version_stdout, stderr=version_stderr)
        raise AssertionError(f"unexpected subprocess.run call: {command!r}")

    class FakeProcess:
        def __init__(self, command, kwargs):
            self.command = list(command)
            self.kwargs = kwargs
            self.pid = 90_000_000 + len(calls.processes)
            self.returncode = None
            self.communicated_input = None
            self.communicate_timeout = None
            self.wait_timeouts = []
            self.terminated = False
            self.killed = False

        def communicate(self, *, input, timeout):
            self.communicated_input = input
            self.communicate_timeout = timeout
            if isinstance(exec_exception, subprocess.TimeoutExpired):
                raise exec_exception
            if exec_exception is not None:
                raise exec_exception
            _write_mock_stream(self.kwargs, "stdout", stdout)
            _write_mock_stream(self.kwargs, "stderr", stderr)
            self.returncode = returncode
            if after_exec is not None:
                after_exec(self.command, self.kwargs)
            return None, None

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.returncode is None:
                self.returncode = -signal.SIGTERM
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.killed = True
            self.returncode = -signal.SIGKILL

    def fake_popen(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        assert "exec" in command, command
        if isinstance(exec_exception, OSError):
            raise exec_exception
        if inspect_exec is not None:
            inspect_exec(command, kwargs)
        process = FakeProcess(command, kwargs)
        calls.processes.append(process)
        return process

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    monkeypatch.setattr(C.subprocess, "Popen", fake_popen)
    return calls


def _run(panel_paths, **kwargs):
    return C.run_codex_structured(
        "solve the attached panels", panel_paths, SIMPLE_SCHEMA,
        model=MODEL, executable="codex", **kwargs)


def test_success_uses_exact_isolated_command_private_view_and_minimal_env(
        panel_paths, monkeypatch):
    observed: dict[str, Any] = {}
    expected_images = {
        os.path.basename(path): open(path, "rb").read()
        for path in panel_paths
    }
    monkeypatch.setenv("CODEX_API_KEY", "codex-auth")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")

    def inspect_exec(command, kwargs):
        view = kwargs["cwd"]
        observed["command"] = command
        observed["kwargs"] = kwargs
        observed["auth_home"] = kwargs["env"]["CODEX_HOME"]
        observed["auth_files"] = sorted(os.listdir(observed["auth_home"]))
        observed["mode"] = stat.S_IMODE(os.stat(view).st_mode)
        observed["files"] = sorted(os.listdir(view))
        observed["images"] = {
            name: open(os.path.join(view, name), "rb").read()
            for name in C._PANEL_NAMES
        }
        with open(os.path.join(view, "output_schema.json"), "rb") as handle:
            observed["schema"] = handle.read()

    calls = _install_cli(
        monkeypatch, _jsonl(), inspect_exec=inspect_exec)
    result = _run(panel_paths)

    assert result.payload == {"answer": "ok"}
    # Fingerprint immediately before and after the turn so replacing the
    # executable under a stable command name cannot stale-label the receipt.
    assert all(call[0][1:] == ["--version"] for call in calls[::2])
    assert calls[0][0][0] == calls[1][0][0] == calls[2][0][0]
    assert len(calls) == 3
    command = observed["command"]
    assert command[1:4] == ["--ask-for-approval", "never", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "--search" not in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == MODEL
    assert command[-1] == "-"
    for flag in (
            "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--strict-config", "--skip-git-repo-check", "--json"):
        assert command.count(flag) == 1
    configs = [command[index + 1] for index, value in enumerate(command[:-1])
               if value in {"--config", "-c"}]
    assert 'model_reasoning_effort="medium"' in configs
    assert 'web_search="disabled"' in configs
    assert "agents.enabled=false" in configs
    disabled = [command[index + 1]
                for index, value in enumerate(command[:-1])
                if value == "--disable"]
    assert disabled == list(C._DISABLED_FEATURES)
    assert set(disabled) == {
        "shell_tool", "unified_exec", "apps", "multi_agent", "hooks",
        "goals", "memories", "remote_plugin",
        "skill_mcp_dependency_install",
        "plugins", "plugin_sharing", "skill_search", "browser_use",
        "browser_use_external", "browser_use_full_cdp_access",
        "computer_use", "image_generation", "in_app_browser",
        "code_mode_host", "auth_elicitation", "tool_call_mcp_elicitation",
        "tool_suggest", "workspace_dependencies", "network_proxy",
        "standalone_web_search",
    }

    assert observed["mode"] == 0o700
    assert observed["files"] == sorted(
        [*C._PANEL_NAMES, "output_schema.json"])
    assert observed["images"] == expected_images
    assert observed["schema"] == C._canonical_json_bytes(SIMPLE_SCHEMA)
    image_args = [command[index + 1]
                  for index, value in enumerate(command[:-1])
                  if value == "--image"]
    assert [os.path.basename(path) for path in image_args] == \
        list(C._PANEL_NAMES)
    assert all(os.path.dirname(path) == observed["kwargs"]["cwd"]
               for path in image_args)
    assert command[command.index("--cd") + 1] == observed["kwargs"]["cwd"]
    assert observed["kwargs"]["stdin"] == subprocess.PIPE
    assert observed["kwargs"]["start_new_session"] is True
    assert calls.processes[0].communicated_input == \
        b"solve the attached panels"
    assert calls.processes[0].communicate_timeout == 15 * 60
    env = observed["kwargs"]["env"]
    assert env["CODEX_API_KEY"] == "codex-auth"
    assert env["CODEX_HOME"] == observed["auth_home"]
    assert env["CODEX_HOME"] != os.environ.get("CODEX_HOME")
    assert observed["auth_files"] == []
    assert env["TERM"] == "dumb" and env["NO_COLOR"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "UNRELATED_SECRET" not in env


def test_success_without_reported_model_is_honest_and_receipt_reproduces(
        panel_paths, monkeypatch):
    stream = _jsonl()
    _install_cli(monkeypatch, stream)
    result = _run(panel_paths, reasoning_effort="high")
    receipt = result.receipt.to_dict()
    assert receipt["requested_model"] == MODEL
    assert receipt["reported_model"] == ""
    assert receipt["model_identity_evidence"] == \
        "explicit-cli-model-flag;jsonl-omits-model"
    effort_key = "requested_reasoning_effort" \
        if "requested_reasoning_effort" in receipt else "reasoning_effort"
    assert receipt[effort_key] == "high"
    assert receipt["input_tokens"] == 17
    assert receipt["cached_input_tokens"] == 3
    assert receipt["output_tokens"] == 5
    assert receipt["reasoning_output_tokens"] == 2
    assert receipt["thread_id"] == THREAD_ID
    assert receipt["codex_cli_version"] == "codex-cli 0.146.0"
    assert receipt["event_stream_digest"] == C._bytes_digest(stream)
    assert receipt["schema"] == "bongard.codex-cli-proposer-receipt/v2"
    assert receipt["input_digest_schema"] == \
        C.STRUCTURED_INPUT_DIGEST_SCHEMA
    expected_task_digest = hashlib.sha256(
        b"solve the attached panels").hexdigest()
    assert receipt["task_digest"] == expected_task_digest
    assert receipt["prompt_digest"] == expected_task_digest
    assert receipt["current_source_digest"] == ""
    assert receipt["current_log_digest"] == ""
    assert receipt["proposed_source_digest"] == ""
    assert receipt["proposed_log_digest"] == ""
    assert receipt["panel_view_digest"] == \
        C.ordered_panel_view_digest(panel_paths)
    assert receipt["panel_set_digest"] == \
        C.semantic_panel_set_digest(panel_paths)
    assert receipt["structured_output_digest"] == \
        C._structured_payload_digest({"answer": "ok"})
    C.validate_codex_receipt(receipt)


def test_named_image_turn_uses_neutral_name_and_distinct_receipt_schema(
        panel_paths, monkeypatch):
    observed: dict[str, Any] = {}

    def inspect_exec(command, kwargs):
        observed["files"] = sorted(os.listdir(kwargs["cwd"]))
        observed["images"] = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--image"
        ]

    _install_cli(monkeypatch, _jsonl(), inspect_exec=inspect_exec)
    result = C.run_codex_named_images_structured(
        "score one unlabeled image",
        [panel_paths[0]],
        ["panel.png"],
        SIMPLE_SCHEMA,
        model=MODEL,
        executable="codex",
    )

    assert result.payload == {"answer": "ok"}
    assert observed["files"] == ["output_schema.json", "panel.png"]
    assert [os.path.basename(path) for path in observed["images"]] == [
        "panel.png"]
    receipt = result.receipt.to_dict()
    assert receipt["input_digest_schema"] == \
        C.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
    assert receipt["panel_view_digest"] == C.named_image_view_digest(
        [panel_paths[0]], ["panel.png"])
    assert receipt["panel_set_digest"] == C.named_image_set_digest(
        [panel_paths[0]], ["panel.png"])
    C.validate_codex_receipt(receipt)


@pytest.mark.parametrize(
    "name", ["pos_0.png", "neg_0.png", "positive.png", "../panel.png"])
def test_named_image_turn_rejects_label_bearing_or_unsafe_names(
        panel_paths, name):
    with pytest.raises(C.CodexProposerFailure, match="name|side"):
        C._named_image_snapshot([panel_paths[0]], [name])


def test_matching_reported_model_is_preserved(panel_paths, monkeypatch):
    _install_cli(monkeypatch, _jsonl(reported_model=MODEL))
    receipt = _run(panel_paths).receipt.to_dict()
    assert receipt["reported_model"] == MODEL
    assert receipt["model_identity_evidence"] == "jsonl-reported-model"


def test_reported_model_substitution_is_rejected(panel_paths, monkeypatch):
    _install_cli(monkeypatch, _jsonl(reported_model="gpt-substitute"))
    with pytest.raises(C.CodexProposerFailure, match="different"):
        _run(panel_paths)


@pytest.mark.parametrize("malformed_model", [None, 17, False, [MODEL]])
def test_present_malformed_model_evidence_is_rejected(
        panel_paths, monkeypatch, malformed_model):
    events = [json.loads(line) for line in _jsonl().decode().splitlines()]
    events[0]["model"] = malformed_model
    stream = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    _install_cli(monkeypatch, stream)
    with pytest.raises(C.CodexProposerFailure, match="malformed model"):
        _run(panel_paths)


def _with_final_message_text(stream: bytes, text: str) -> bytes:
    events = [json.loads(line) for line in stream.decode().splitlines()]
    message = next(
        event for event in events
        if event["type"] == "item.completed"
        and event["item"]["type"] == "agent_message")
    message["item"]["text"] = text
    return ("\n".join(json.dumps(event, separators=(",", ":"))
                       for event in events) + "\n").encode()


@pytest.mark.parametrize("stream", [
    b"{malformed\n",
    b"\xff\n",
    _jsonl().replace(
        b'{"type":"thread.started",',
        b'{"type":"thread.started","type":"thread.started",', 1),
    _jsonl().replace(b'"input_tokens":17', b'"input_tokens":NaN', 1),
    _jsonl().replace(b"\n{\"type\":\"turn.started\"}",
                     b"\n\n{\"type\":\"turn.started\"}", 1),
])
def test_jsonl_rejects_malformed_duplicate_nonfinite_and_bad_framing(
        panel_paths, monkeypatch, stream):
    _install_cli(monkeypatch, stream)
    with pytest.raises(C.CodexProposerFailure):
        _run(panel_paths)


@pytest.mark.parametrize("message", [
    "not-json",
    "[]",
    '{"answer":"first","answer":"duplicate"}',
    '{"answer":NaN}',
])
def test_final_message_rejects_nonobject_duplicate_and_nonfinite_json(
        panel_paths, monkeypatch, message):
    _install_cli(monkeypatch, _with_final_message_text(_jsonl(), message))
    with pytest.raises(C.CodexProposerFailure):
        _run(panel_paths)


@pytest.mark.parametrize("event", [
    {"type": "error", "message": "provider failed"},
    {"type": "turn.failed", "error": "provider failed"},
    {"type": "unexpected.future.event"},
])
def test_failed_error_and_unknown_events_are_rejected(
        panel_paths, monkeypatch, event):
    _install_cli(monkeypatch, _jsonl(extra_events=[event]))
    with pytest.raises(C.CodexProposerFailure):
        _run(panel_paths)


@pytest.mark.parametrize("item_type", [
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
    "collab_tool_call",
    "view_image",
])
def test_every_command_file_mcp_web_and_other_tool_item_is_rejected(
        panel_paths, monkeypatch, item_type):
    tool_event = {
        "type": "item.completed",
        "item": {"id": "forbidden", "type": item_type, "text": "done"},
    }
    _install_cli(monkeypatch, _jsonl(extra_events=[tool_event]))
    with pytest.raises(C.CodexProposerFailure, match="forbidden|unsupported"):
        _run(panel_paths)


def test_a_completed_final_agent_message_is_required(panel_paths, monkeypatch):
    _install_cli(monkeypatch, _jsonl(messages=0))
    with pytest.raises(
            C.CodexProposerFailure,
            match="completed final"):
        _run(panel_paths)


def test_completed_progress_messages_preceding_final_are_receipted(
        panel_paths, monkeypatch):
    stream = _jsonl(messages=2)
    _install_cli(monkeypatch, stream)
    result = _run(panel_paths)
    assert result.payload == {"answer": "ok"}
    assert result.receipt.item_types == ("agent_message", "agent_message")
    assert result.receipt.event_stream_digest == C._bytes_digest(stream)


def test_agent_message_must_be_completed_text(panel_paths, monkeypatch):
    event = {
        "type": "item.started",
        "item": {"id": "draft", "type": "agent_message"},
    }
    _install_cli(monkeypatch, _jsonl(messages=0, extra_events=[event]))
    with pytest.raises(
            C.CodexProposerFailure,
            match="completed final"):
        _run(panel_paths)


def test_reasoning_after_final_agent_message_is_rejected(
        panel_paths, monkeypatch):
    events = [json.loads(line) for line in _jsonl().decode().splitlines()]
    events.insert(-1, {
        "type": "item.completed",
        "item": {"id": "late_reasoning", "type": "reasoning",
                 "text": "too late"},
    })
    stream = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    _install_cli(monkeypatch, stream)
    with pytest.raises(C.CodexProposerFailure, match="final item"):
        _run(panel_paths)


@pytest.mark.parametrize("thread_id", [
    "not-a-uuid",
    THREAD_ID.upper(),
    "",
])
def test_thread_id_must_be_canonical_uuid(
        panel_paths, monkeypatch, thread_id):
    _install_cli(monkeypatch, _jsonl(thread_id=thread_id))
    with pytest.raises(C.CodexProposerFailure, match="thread ID"):
        _run(panel_paths)


@pytest.mark.parametrize("usage", [
    {"input_tokens": 1},
    {
        "input_tokens": -1, "cached_input_tokens": 0,
        "output_tokens": 1, "reasoning_output_tokens": 0,
    },
    {
        "input_tokens": True, "cached_input_tokens": 0,
        "output_tokens": 1, "reasoning_output_tokens": 0,
    },
    {
        "input_tokens": 1, "cached_input_tokens": 2,
        "output_tokens": 1, "reasoning_output_tokens": 0,
    },
    {
        "input_tokens": 0, "cached_input_tokens": 0,
        "output_tokens": 0, "reasoning_output_tokens": 0,
    },
])
def test_usage_must_be_complete_nonnegative_positive_and_consistent(
        panel_paths, monkeypatch, usage):
    _install_cli(monkeypatch, _jsonl(usage=usage))
    with pytest.raises(C.CodexProposerFailure, match="usage|tokens"):
        _run(panel_paths)


def test_reordered_lifecycle_is_rejected(panel_paths, monkeypatch):
    events = [json.loads(line) for line in _jsonl().decode().splitlines()]
    events[1], events[2] = events[2], events[1]
    stream = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    _install_cli(monkeypatch, stream)
    with pytest.raises(C.CodexProposerFailure, match="forbidden|lifecycle"):
        _run(panel_paths)


def test_stdout_and_stderr_are_capped(panel_paths, monkeypatch):
    _install_cli(monkeypatch, b"x" * (C.MAX_STDOUT_BYTES + 1))
    with pytest.raises(C.CodexProposerFailure, match="oversized"):
        _run(panel_paths)

    _install_cli(
        monkeypatch, _jsonl(), returncode=7,
        stderr=b"x" * (C.MAX_STDERR_BYTES + 1))
    with pytest.raises(C.CodexProposerFailure, match="diagnostic.*oversized"):
        _run(panel_paths)


@pytest.mark.parametrize(("field", "maximum"), [
    ("predicates_source", C.MAX_PREDICATE_SOURCE_UTF8_BYTES),
    ("predicates_log", C.MAX_PREDICATE_LOG_UTF8_BYTES),
    ("rationale", C.MAX_RATIONALE_UTF8_BYTES),
])
def test_predicate_returned_strings_are_independently_capped(
        panel_paths, monkeypatch, field, maximum):
    payload = {
        "predicates_source": "# source\n",
        "predicates_log": "",
        "rationale": "because",
    }
    payload[field] = "x" * (maximum + 1)
    _install_cli(monkeypatch, _jsonl(payload))
    with pytest.raises(C.CodexProposerFailure, match="exceeds|oversized"):
        C.run_codex_proposer(
            "task", panel_paths, "# existing\n", "",
            model=MODEL, executable="codex")


def test_predicate_wrapper_returns_complete_values_without_applying_them(
        panel_paths, monkeypatch, tmp_path):
    payload = {
        "predicates_source": "def p_count(panel):\n    return 1.0\n",
        "predicates_log": "count primitive\n",
        "rationale": "one reusable measurement",
    }
    workspace_sentinel = tmp_path / "predicates.py"
    workspace_sentinel.write_text("untouched")
    _install_cli(monkeypatch, _jsonl(payload))
    task = "scientific contract"
    current_source = "# existing\n"
    current_log = "old log\n"
    result = C.run_codex_proposer(
        task, panel_paths, current_source, current_log,
        model=MODEL, executable="codex")
    assert result.predicates_source == payload["predicates_source"]
    assert result.predicates_log == payload["predicates_log"]
    assert result.rationale == payload["rationale"]
    receipt = result.receipt.to_dict()
    assert receipt["input_digest_schema"] == C.PREDICATE_INPUT_DIGEST_SCHEMA
    assert receipt["task_digest"] == hashlib.sha256(task.encode()).hexdigest()
    assert receipt["current_source_digest"] == hashlib.sha256(
        current_source.encode()).hexdigest()
    assert receipt["current_log_digest"] == hashlib.sha256(
        current_log.encode()).hexdigest()
    assert receipt["prompt_digest"] == hashlib.sha256(
        C._predicate_prompt(task, current_source, current_log).encode()
    ).hexdigest()
    assert receipt["input_digest"] == C.predicate_proposer_input_digest(
        task, current_source, current_log, panel_paths)
    assert receipt["structured_output_digest"] == \
        C.predicate_proposer_output_digest(**payload)
    assert receipt["proposed_source_digest"] == hashlib.sha256(
        payload["predicates_source"].encode()).hexdigest()
    assert receipt["proposed_log_digest"] == hashlib.sha256(
        payload["predicates_log"].encode()).hexdigest()
    assert receipt["panel_set_digest"] == \
        C.semantic_panel_set_digest(panel_paths)
    assert workspace_sentinel.read_text() == "untouched"


def test_public_panel_digests_bind_raw_pngs_and_semantic_panel_set(
        panel_paths, tmp_path):
    records = []
    for path in panel_paths:
        name = os.path.basename(path)
        side, raw_index = name[:-4].split("_")
        with Image.open(path) as encoded:
            presentation = np.asarray(encoded.convert("L"), dtype=np.uint8)
        panel = (presentation == 0).astype(np.uint8)
        records.append(SR.PanelRecord.from_array(
            panel, side, int(raw_index)))
    expected_semantic = SR.panel_set_digest(tuple(records))
    assert C.semantic_panel_set_digest(panel_paths) == expected_semantic

    alternate = tmp_path / "alternate"
    alternate.mkdir()
    alternate_paths = []
    for path in panel_paths:
        target = alternate / os.path.basename(path)
        with Image.open(path) as encoded:
            presentation = np.asarray(encoded.convert("L"), dtype=np.uint8)
        if target.name == "pos_0.png":
            from PIL.PngImagePlugin import PngInfo
            metadata = PngInfo()
            metadata.add_text("causal-test", "same pixels, different PNG bytes")
            Image.fromarray(presentation, mode="L").save(
                target, format="PNG", pnginfo=metadata)
        else:
            Image.fromarray(presentation, mode="L").save(target, format="PNG")
        alternate_paths.append(str(target.resolve()))
    assert C.ordered_panel_view_digest(alternate_paths) != \
        C.ordered_panel_view_digest(panel_paths)
    assert C.semantic_panel_set_digest(alternate_paths) == expected_semantic


def test_public_predicate_input_and_output_digests_change_with_every_cause(
        panel_paths, tmp_path):
    task = "task"
    source = "# source\n"
    log = "log\n"
    baseline_input = C.predicate_proposer_input_digest(
        task, source, log, panel_paths)
    changed_inputs = {
        C.predicate_proposer_input_digest(
            task + "!", source, log, panel_paths),
        C.predicate_proposer_input_digest(
            task, source + "# changed\n", log, panel_paths),
        C.predicate_proposer_input_digest(
            task, source, log + "changed\n", panel_paths),
    }
    assert baseline_input not in changed_inputs
    assert len(changed_inputs) == 3

    changed_dir = tmp_path / "changed-panels"
    changed_dir.mkdir()
    changed_paths = []
    for path in panel_paths:
        target = changed_dir / os.path.basename(path)
        target.write_bytes(open(path, "rb").read())
        changed_paths.append(str(target.resolve()))
    with Image.open(changed_paths[0]) as encoded:
        presentation = np.asarray(encoded.convert("L"), dtype=np.uint8).copy()
    presentation[15, 15] = 0
    Image.fromarray(presentation, mode="L").save(
        changed_paths[0], format="PNG")
    assert C.semantic_panel_set_digest(changed_paths) != \
        C.semantic_panel_set_digest(panel_paths)
    assert C.predicate_proposer_input_digest(
        task, source, log, changed_paths) != baseline_input

    baseline_output = C.predicate_proposer_output_digest(
        "source", "log", "rationale")
    changed_outputs = {
        C.predicate_proposer_output_digest("source!", "log", "rationale"),
        C.predicate_proposer_output_digest("source", "log!", "rationale"),
        C.predicate_proposer_output_digest("source", "log", "rationale!"),
    }
    assert baseline_output not in changed_outputs
    assert len(changed_outputs) == 3


def test_predicate_wrapper_rejects_missing_extra_and_nonstring_fields(
        panel_paths, monkeypatch):
    payloads = [
        {"predicates_source": "x", "predicates_log": ""},
        {
            "predicates_source": "x", "predicates_log": "",
            "rationale": "r", "extra": "no",
        },
        {
            "predicates_source": 3, "predicates_log": "",
            "rationale": "r",
        },
    ]
    for payload in payloads:
        _install_cli(monkeypatch, _jsonl(payload))
        with pytest.raises(C.CodexProposerFailure):
            C.run_codex_proposer(
                "task", panel_paths, "# source\n", "",
                model=MODEL, executable="codex")


def test_input_task_current_library_log_and_schema_are_capped(
        panel_paths):
    with pytest.raises(C.CodexProposerFailure, match="task.*exceeds"):
        _run(panel_paths, task="unused") if False else \
            C.run_codex_structured(
                "x" * (C.MAX_TASK_UTF8_BYTES + 1), panel_paths,
                SIMPLE_SCHEMA, model=MODEL)
    with pytest.raises(C.CodexProposerFailure, match="source.*exceeds"):
        C.run_codex_proposer(
            "task", panel_paths,
            "x" * (C.MAX_PREDICATE_SOURCE_UTF8_BYTES + 1), "",
            model=MODEL)
    with pytest.raises(C.CodexProposerFailure, match="log.*exceeds"):
        C.run_codex_proposer(
            "task", panel_paths, "# source\n",
            "x" * (C.MAX_PREDICATE_LOG_UTF8_BYTES + 1), model=MODEL)
    huge_schema = {"description": "x" * C.MAX_SCHEMA_UTF8_BYTES}
    with pytest.raises(C.CodexProposerFailure, match="schema.*oversized"):
        C.run_codex_structured(
            "task", panel_paths, huge_schema, model=MODEL)


@pytest.mark.parametrize("paths_transform", [
    lambda paths: paths[:-1],
    lambda paths: paths[:-1] + [paths[0]],
])
def test_panel_view_requires_exact_canonical_twelve(
        panel_paths, monkeypatch, paths_transform):
    _install_cli(monkeypatch, _jsonl())
    with pytest.raises(C.CodexProposerFailure, match="12|filenames"):
        _run(paths_transform(panel_paths))


def test_panel_view_rejects_relative_symlink_non_png_and_oversized(
        panel_paths, tmp_path, monkeypatch):
    _install_cli(monkeypatch, _jsonl())

    relative = list(panel_paths)
    relative[0] = "pos_0.png"
    with pytest.raises(C.CodexProposerFailure, match="absolute"):
        _run(relative)

    target = tmp_path / "real.png"
    target.write_bytes(open(panel_paths[0], "rb").read())
    link = tmp_path / "pos_0.png.link"
    link.symlink_to(target)
    symlinked = list(panel_paths)
    # The canonical basename is part of the contract, so make a canonical
    # link in a separate directory.
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    canonical_link = link_dir / "pos_0.png"
    canonical_link.symlink_to(target)
    symlinked[0] = str(canonical_link.resolve(strict=False))
    # pathlib.resolve follows links; retain the symlink path explicitly.
    symlinked[0] = str(canonical_link.absolute())
    with pytest.raises(
            C.CodexProposerFailure,
            match="cannot open|not a bounded.*regular file"):
        _run(symlinked)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = bad_dir / "pos_0.png"
    bad.write_bytes(b"not-png")
    invalid = list(panel_paths)
    invalid[0] = str(bad.resolve())
    with pytest.raises(C.CodexProposerFailure, match="not a PNG"):
        _run(invalid)

    large_dir = tmp_path / "large"
    large_dir.mkdir()
    large = large_dir / "pos_0.png"
    large.write_bytes(PNG_SIGNATURE + b"x" * C.MAX_PANEL_PNG_BYTES)
    oversized = list(panel_paths)
    oversized[0] = str(large.resolve())
    with pytest.raises(C.CodexProposerFailure, match="bounded|oversized"):
        _run(oversized)


@pytest.mark.parametrize("target_name", ["pos_0.png", "output_schema.json"])
def test_same_size_private_input_mutation_after_process_is_rejected(
        panel_paths, monkeypatch, target_name):
    def mutate_after_turn(_command, kwargs):
        target = os.path.join(kwargs["cwd"], target_name)
        with open(target, "r+b") as handle:
            data = bytearray(handle.read())
            assert data
            index = -1
            data[index] ^= 1
            handle.seek(0)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        assert os.path.getsize(target) == len(data)

    _install_cli(monkeypatch, _jsonl(), after_exec=mutate_after_turn)
    expected = "panel view|output schema"
    with pytest.raises(C.CodexProposerFailure, match=expected):
        _run(panel_paths)


def test_hostile_temp_parent_inside_workspace_is_rejected_before_creation(
        panel_paths, monkeypatch):
    created = []

    def forbidden_temporary_directory(*args, **kwargs):
        created.append((args, kwargs))
        raise AssertionError("temporary directory must not be created")

    monkeypatch.setenv("TMPDIR", C.WORKSPACE_ROOT)
    monkeypatch.setattr(
        C.tempfile, "TemporaryDirectory", forbidden_temporary_directory)
    with pytest.raises(C.CodexProposerFailure, match="outside the workspace"):
        _run(panel_paths)
    assert created == []


def test_nonzero_timeout_and_launcher_failure_are_infrastructure_failures(
        panel_paths, monkeypatch):
    _install_cli(
        monkeypatch, b"", returncode=17, stderr=b"authentication failed")
    with pytest.raises(C.CodexProposerFailure, match="exited 17"):
        _run(panel_paths)

    group_signals = []
    monkeypatch.setattr(
        C.os, "killpg",
        lambda process_group, sig: group_signals.append((process_group, sig)))
    timeout_calls = _install_cli(
        monkeypatch, b"",
        exec_exception=subprocess.TimeoutExpired(["codex"], timeout=60))
    with pytest.raises(C.CodexProposerFailure, match="timed out"):
        _run(panel_paths)
    timed_out = timeout_calls.processes[0]
    assert group_signals == [
        (timed_out.pid, signal.SIGTERM),
        (timed_out.pid, signal.SIGKILL),
    ]
    assert timed_out.wait_timeouts == [
        C.PROCESS_GROUP_GRACE_SECONDS,
        C.PROCESS_GROUP_GRACE_SECONDS,
    ]

    _install_cli(monkeypatch, b"", exec_exception=OSError("not found"))
    with pytest.raises(C.CodexProposerFailure, match="could not be launched"):
        _run(panel_paths)


def test_timeout_cleanup_has_portable_process_fallback(monkeypatch):
    class StubbornProcess:
        pid = 91_000_000
        returncode = None
        terminated = False
        killed = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.returncode = -signal.SIGKILL

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(["codex"], timeout=timeout)
            return self.returncode

    monkeypatch.setattr(
        C.os, "killpg",
        lambda _group, _signal: (_ for _ in ()).throw(
            OSError("groups unsupported")))
    process = StubbornProcess()
    C._terminate_process_group(process)
    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == -signal.SIGKILL


def test_reasoning_model_timeout_and_schema_inputs_are_validated(panel_paths):
    with pytest.raises(C.CodexProposerFailure, match="model identifier"):
        C.run_codex_structured("task", panel_paths, SIMPLE_SCHEMA, model="")
    with pytest.raises(C.CodexProposerFailure, match="reasoning effort"):
        C.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, model=MODEL,
            reasoning_effort="adaptive")
    with pytest.raises(C.CodexProposerFailure, match="timeout"):
        C.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, model=MODEL, minutes=0)
    with pytest.raises(C.CodexProposerFailure, match="mapping"):
        C.run_codex_structured(
            "task", panel_paths, [], model=MODEL)  # type: ignore[arg-type]


def _redigest(receipt):
    body = {key: value for key, value in receipt.items()
            if key != "receipt_digest"}
    receipt["receipt_digest"] = C._digest(body)


def test_receipt_digest_and_semantics_reject_tampering(
        panel_paths, monkeypatch):
    _install_cli(monkeypatch, _jsonl())
    original = _run(panel_paths).receipt.to_dict()

    tampered = copy.deepcopy(original)
    tampered["input_tokens"] += 1
    with pytest.raises(C.CodexProposerFailure, match="digest"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["input_digest"] = "0" * 64
    with pytest.raises(C.CodexProposerFailure, match="digest"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["input_digest_schema"] = "unknown-input/v1"
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="input digest schema"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["panel_set_digest"] = "0" * 64
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="panel_set_digest"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["structured_output_digest"] = "not-a-digest"
    _redigest(tampered)
    with pytest.raises(
            C.CodexProposerFailure,
            match="structured_output_digest"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["current_source_digest"] = "1" * 64
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="generic receipt"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["event_types"].insert(-1, "item.completed")
    tampered["item_types"].append("command_execution")
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="item summary"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["event_types"].insert(-1, "web_search")
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="event summary"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["model_identity_evidence"] = "provider-attested"
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="evidence"):
        C.validate_codex_receipt(tampered)

    tampered = copy.deepcopy(original)
    tampered["thread_id"] = THREAD_ID.upper()
    _redigest(tampered)
    with pytest.raises(C.CodexProposerFailure, match="non-canonical"):
        C.validate_codex_receipt(tampered)
