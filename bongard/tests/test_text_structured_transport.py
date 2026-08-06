"""Offline tests for the zero-image structured Codex transport.

The subprocess boundary is fully faked.  No Codex/model or shell process is
started by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import pytest

import bongard.transport as T


THREAD_ID = "12345678-1234-4abc-9234-abcdef123456"
SCHEMA = {
    "type": "object",
    "properties": {
        "atoms": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["atoms"],
    "additionalProperties": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _jsonl() -> bytes:
    events = [
        {
            "type": "thread.started",
            "thread_id": THREAD_ID,
            "model": "gpt-5.6-sol",
        },
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": '{"atoms":["acute corners"]}',
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 19,
                "cached_input_tokens": 4,
                "output_tokens": 7,
                "reasoning_output_tokens": 3,
            },
        },
    ]
    return b"\n".join(_canonical(event) for event in events) + b"\n"


def _fake_launcher(tmp_path: Path) -> str:
    launcher = tmp_path / "fake-codex"
    launcher.write_bytes(b"not executed")
    launcher.chmod(0o700)
    return str(launcher.resolve())


def test_text_structured_turn_binds_zero_images_schema_auth_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "source-codex-home"
    source_home.mkdir()
    auth_bytes = b'{"tokens":{"access_token":"offline-test"}}\n'
    cache_bytes = (
        b'{"signed_payload":{"policy":"offline"},'
        b'"signature":"offline-signature"}\n'
    )
    (source_home / "auth.json").write_bytes(auth_bytes)
    (source_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(cache_bytes)
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    launcher = _fake_launcher(tmp_path)
    prompt = "Extract atomic affirmative predicates from these descriptions."
    observed: dict[str, Any] = {"version_calls": 0, "process_calls": 0}

    def fake_run(command, **kwargs):
        command = list(command)
        assert command == [launcher, "--version"]
        observed["version_calls"] += 1
        kwargs["stdout"].write(b"codex-cli 0.offline\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    class FakeProcess:
        pid = 91_000_001

        def __init__(self, command: list[str], kwargs: dict[str, Any]):
            self.command = command
            self.kwargs = kwargs
            self.returncode: int | None = None

        def communicate(self, *, input: bytes, timeout: int):
            observed["stdin"] = input
            observed["timeout"] = timeout
            self.kwargs["stdout"].write(_jsonl())
            self.kwargs["stdout"].flush()
            self.returncode = 0
            return None, None

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **kwargs):
        command = list(command)
        observed["process_calls"] += 1
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["environment"] = dict(kwargs["env"])
        view = Path(kwargs["cwd"])
        observed["view_mode"] = stat.S_IMODE(view.stat().st_mode)
        observed["view_files"] = sorted(path.name for path in view.iterdir())
        observed["schema_bytes"] = (view / "output_schema.json").read_bytes()
        auth_home = Path(kwargs["env"]["CODEX_HOME"])
        observed["auth_files"] = sorted(path.name for path in auth_home.iterdir())
        observed["auth_bytes"] = (auth_home / "auth.json").read_bytes()
        observed["cache_bytes"] = (
            auth_home / T._CLOUD_CONFIG_BUNDLE_CACHE
        ).read_bytes()
        return FakeProcess(command, kwargs)

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)

    result = T.run_codex_text_structured(
        prompt,
        SCHEMA,
        reasoning_effort="high",
        minutes=2,
        executable=launcher,
    )

    assert result.payload == {"atoms": ["acute corners"]}
    assert observed["version_calls"] == 2
    assert observed["process_calls"] == 1
    assert observed["stdin"] == prompt.encode("utf-8")
    assert observed["timeout"] == 120
    assert observed["view_mode"] == 0o700
    assert observed["view_files"] == ["output_schema.json"]
    assert observed["schema_bytes"] == _canonical(SCHEMA)
    assert observed["auth_files"] == [
        "auth.json", T._CLOUD_CONFIG_BUNDLE_CACHE
    ]
    assert observed["auth_bytes"] == auth_bytes
    assert observed["cache_bytes"] == cache_bytes
    assert os.path.commonpath((observed["cwd"], T.WORKSPACE_ROOT)) != (
        T.WORKSPACE_ROOT
    )

    command = observed["command"]
    assert "--image" not in command
    assert command[-1] == "-"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--output-schema") + 1] == str(
        Path(observed["cwd"]) / "output_schema.json"
    )
    assert command[command.index("--cd") + 1] == observed["cwd"]
    assert 'model_reasoning_effort="high"' in command
    disabled = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--disable"
    ]
    assert disabled == list(T._DISABLED_FEATURES)

    schema_digest = hashlib.sha256(_canonical(SCHEMA)).hexdigest()
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    zero_view_digest = _digest([])
    zero_set_digest = "sha256:" + _digest({
        "schema": T.TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
        "images": [],
    })
    expected_envelope = {
        "schema": T.TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "image_count": 0,
        "image_view_digest": zero_view_digest,
        "image_set_digest": zero_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    receipt = result.receipt.to_dict()
    assert receipt["source"] == "codex-cli"
    assert receipt["input_digest_schema"] == (
        T.TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA
    )
    assert receipt["task_digest"] == receipt["prompt_digest"] == prompt_digest
    assert receipt["output_schema_digest"] == schema_digest
    assert receipt["input_digest"] == _digest(expected_envelope)
    assert receipt["panel_view_digest"] == zero_view_digest
    assert receipt["panel_set_digest"] == zero_set_digest
    assert receipt["cloud_config_bundle_cache_binding"] == (
        "sha256:" + hashlib.sha256(cache_bytes).hexdigest()
    )
    T.validate_codex_receipt(receipt)
    T.validate_codex_text_receipt(receipt, prompt, SCHEMA)

    with pytest.raises(T.CodexProposerFailure, match="supplied prompt and schema"):
        T.validate_codex_text_receipt(receipt, prompt + " altered", SCHEMA)
    other_schema = copy.deepcopy(SCHEMA)
    other_schema["properties"]["atoms"]["items"] = {"type": "integer"}
    with pytest.raises(T.CodexProposerFailure, match="supplied prompt and schema"):
        T.validate_codex_text_receipt(receipt, prompt, other_schema)

    # Generic receipt integrity alone cannot establish the caller's causal
    # domain.  Even a validly re-digested attempt to relabel this receipt as
    # an image-domain receipt is rejected by the text-specific validator.
    cross_domain = copy.deepcopy(receipt)
    cross_domain["input_digest_schema"] = T.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
    cross_domain["receipt_digest"] = _digest({
        key: value
        for key, value in cross_domain.items()
        if key != "receipt_digest"
    })
    T.validate_codex_receipt(cross_domain)
    with pytest.raises(T.CodexProposerFailure, match="text-only input domain"):
        T.validate_codex_text_receipt(cross_domain, prompt, SCHEMA)

    # A validly re-digested receipt still cannot forge an image-bearing text
    # turn: text receipts have one exact zero-image sentinel pair.
    forged = copy.deepcopy(receipt)
    forged["panel_view_digest"] = "0" * 64
    forged["receipt_digest"] = _digest({
        key: value for key, value in forged.items() if key != "receipt_digest"
    })
    with pytest.raises(T.CodexProposerFailure, match="zero-image"):
        T.validate_codex_receipt(forged)


def test_text_structured_turn_rejects_non_strict_schema_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "source-codex-home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    launcher = _fake_launcher(tmp_path)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid schema must fail before subprocess use")

    monkeypatch.setattr(T.subprocess, "run", forbidden)
    monkeypatch.setattr(T.subprocess, "Popen", forbidden)
    invalid_schema = {
        "type": "object",
        "properties": {
            "atoms": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
        "required": ["atoms"],
        "additionalProperties": False,
    }

    with pytest.raises(
        T.CodexProposerFailure,
        match="unsupported keywords: minItems",
    ):
        T.run_codex_text_structured(
            "prompt", invalid_schema, executable=launcher
        )
    assert calls == []
