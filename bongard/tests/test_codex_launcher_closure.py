"""Offline tests for the Codex npm-wrapper executable closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

import bongard.transport as T


def _native_prefix(system: str) -> bytes:
    return {
        "linux": b"\x7fELF\x02\x01\x01\x00",
        "darwin": b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01",
        "win32": b"MZ\x90\x00\x03\x00\x00\x00",
    }[system]


def _official_wrapper_tree(tmp_path: Path) -> tuple[Path, Path]:
    target, platform_package, npm_os, npm_cpu = T._codex_platform_package()
    package_root = tmp_path / "node_modules" / "@openai" / "codex"
    js_entrypoint = package_root / "bin" / "codex.js"
    js_entrypoint.parent.mkdir(parents=True)
    js_entrypoint.write_bytes(
        b"#!/usr/bin/env node\n// fixture entrypoint; never executed\n")
    js_entrypoint.chmod(0o700)
    root_manifest = {
        "name": "@openai/codex",
        "version": "1.2.3",
        "bin": {"codex": "bin/codex.js"},
        "type": "module",
        "optionalDependencies": {platform_package: "fixture"},
    }
    (package_root / "package.json").write_text(
        json.dumps(root_manifest), encoding="utf-8")

    scope, platform_name = platform_package.split("/", 1)
    platform_root = package_root / "node_modules" / scope / platform_name
    platform_root.mkdir(parents=True)
    platform_manifest = {
        "name": "@openai/codex",
        "version": "1.2.3-fixture",
        "os": [npm_os],
        "cpu": [npm_cpu],
    }
    (platform_root / "package.json").write_text(
        json.dumps(platform_manifest), encoding="utf-8")
    native_name = "codex.exe" if npm_os == "win32" else "codex"
    native = platform_root / "vendor" / target / "bin" / native_name
    native.parent.mkdir(parents=True)
    native.write_bytes(_native_prefix(npm_os) + b"native-fixture-v1")
    native.chmod(0o700)
    return js_entrypoint.resolve(), native.resolve()


def _fake_version_run(
        expected_executable: Path,
        calls: list[list[str]],
        *, after: Any = None, output: bytes = b"codex-cli 1.2.3\n"):
    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        assert command == [str(expected_executable), "--version"]
        kwargs["stdout"].write(output)
        kwargs["stdout"].flush()
        if after is not None:
            after()
        return subprocess.CompletedProcess(command, 0)

    return fake_run


def _fake_staged_version_run(
        calls: list[list[str]], *, output: bytes = b"codex-cli 1.2.3\n",
        after: Any = None):
    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        assert len(command) == 2
        assert command[1] == "--version"
        assert Path(command[0]).parent.name.startswith(
            "bongard-codex-launcher-")
        kwargs["stdout"].write(output)
        kwargs["stdout"].flush()
        if after is not None:
            after()
        return subprocess.CompletedProcess(command, 0)

    return fake_run


def _structured_jsonl() -> bytes:
    events = [
        {
            "type": "thread.started",
            "thread_id": "12345678-1234-4abc-9234-abcdef123456",
            "model": T.DEFAULT_CODEX_MODEL,
        },
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": '{"answer":"ok"}',
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 4,
                "cached_input_tokens": 0,
                "output_tokens": 2,
                "reasoning_output_tokens": 1,
            },
        },
    ]
    return (
        "\n".join(json.dumps(event, separators=(",", ":")) for event in events)
        + "\n"
    ).encode("utf-8")


def test_official_js_entrypoint_hashes_and_executes_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess, "run", _fake_version_run(native, calls))

    fingerprint = T.codex_cli_fingerprint(str(wrapper))

    assert calls == [[str(native), "--version"]]
    assert fingerprint == {
        "version": "codex-cli 1.2.3",
        "launcher_digest": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    assert fingerprint["launcher_digest"] != hashlib.sha256(
        wrapper.read_bytes()).hexdigest()
    resolved, identity = T._codex_launcher_identity(str(wrapper))
    command = T._codex_command(
        executable=resolved,
        view_dir=str(tmp_path),
        image_paths=(),
        schema_path=str(tmp_path / "schema.json"),
        model_catalog_path=str(tmp_path / "model_catalog.json"),
        model=T.DEFAULT_CODEX_MODEL,
        reasoning_effort=T.DEFAULT_REASONING_EFFORT,
    )
    assert command[0] == str(native)
    assert identity[-1] == fingerprint["launcher_digest"]


def test_structured_receipt_binds_native_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    expected_native_digest = hashlib.sha256(native.read_bytes()).hexdigest()
    source_home = tmp_path / "source-codex-home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    version_calls: list[list[str]] = []
    process_commands: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess,
        "run",
        _fake_version_run(
            native,
            version_calls,
            output=(T.PINNED_CODEX_CLI_VERSION + "\n").encode("utf-8"),
        ),
    )
    monkeypatch.setattr(
        T,
        "_resolve_no_tools_attestation",
        lambda **_kwargs: "a" * 64,
    )

    def fake_process(command, **kwargs):
        del kwargs
        process_commands.append(list(command))
        return 0, _structured_jsonl(), b""

    monkeypatch.setattr(T, "_run_codex_process", fake_process)
    result = T.run_codex_text_structured(
        "fixture prompt",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        executable=str(wrapper),
        expected_launcher_digest=expected_native_digest,
    )

    assert result.payload == {"answer": "ok"}
    assert result.receipt.codex_launcher_digest == expected_native_digest
    assert version_calls == [
        [str(native), "--version"], [str(native), "--version"]]
    assert len(process_commands) == 1
    assert process_commands[0][0] == str(native)


def test_authenticated_fingerprint_executes_only_the_staged_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    expected_native_digest = hashlib.sha256(native.read_bytes()).hexdigest()
    wrapper_before = wrapper.read_bytes()
    calls: list[list[str]] = []

    def mutate_native() -> None:
        old = native.read_bytes()
        assert old.endswith(b"v1")
        native.write_bytes(old[:-2] + b"v2")

    monkeypatch.setattr(
        T.subprocess,
        "run",
        _fake_staged_version_run(calls, after=mutate_native),
    )
    fingerprint = T.codex_cli_authenticated_fingerprint(
        str(wrapper),
        expected_launcher_digest=expected_native_digest,
    )

    assert fingerprint == {
        "version": "codex-cli 1.2.3",
        "launcher_digest": expected_native_digest,
    }
    assert len(calls) == 1
    assert calls[0][0] != str(native)
    assert not Path(calls[0][0]).exists()
    assert wrapper.read_bytes() == wrapper_before
    assert hashlib.sha256(native.read_bytes()).hexdigest() != (
        expected_native_digest)


def test_stage_copies_one_open_source_fd_and_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    resolved, identity = T._codex_launcher_identity(str(wrapper))
    assert resolved == str(native)
    saved_native = native.with_name("codex-original")
    real_read = T.os.read
    swapped = False

    def swapping_read(descriptor: int, maximum: int) -> bytes:
        nonlocal swapped
        block = real_read(descriptor, maximum)
        if block and not swapped:
            swapped = True
            native.rename(saved_native)
            native.write_bytes(_native_prefix(T._codex_platform_package()[2])
                               + b"attacker-substitute")
            native.chmod(0o700)
        return block

    monkeypatch.setattr(
        T, "_codex_launcher_identity", lambda executable: (resolved, identity))
    monkeypatch.setattr(T.os, "read", swapping_read)

    with pytest.raises(
        T.CodexProposerFailure,
        match="launcher changed while being staged",
    ):
        with T.stage_codex_launcher(
                str(wrapper),
                expected_launcher_digest=identity[-1]):
            raise AssertionError("a swapped source path must not be yielded")
    assert swapped


def test_official_launcher_stages_private_read_execute_only_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    native_bytes = native.read_bytes()
    expected_digest = hashlib.sha256(native_bytes).hexdigest()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess, "run", _fake_staged_version_run(calls))

    with T.stage_codex_launcher(
            str(wrapper),
            expected_launcher_digest=expected_digest) as staged:
        staged_path = Path(staged.executable)
        assert isinstance(staged, T.StagedCodexLauncher)
        assert staged.launcher_digest == expected_digest
        assert staged.version == "codex-cli 1.2.3"
        assert staged_path != native
        assert staged_path.read_bytes() == native_bytes
        assert staged_path.stat().st_mode & 0o777 == 0o500
        assert staged_path.parent.stat().st_mode & 0o777 == 0o500
        assert calls == [[str(staged_path), "--version"]]

    assert not staged_path.exists()


def test_staged_path_is_used_for_every_structured_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    expected_digest = hashlib.sha256(native.read_bytes()).hexdigest()
    source_home = tmp_path / "source-codex-home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    version_calls: list[list[str]] = []
    process_commands: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess,
        "run",
        _fake_staged_version_run(
            version_calls,
            output=(T.PINNED_CODEX_CLI_VERSION + "\n").encode("utf-8"),
        ),
    )
    monkeypatch.setattr(
        T,
        "_resolve_no_tools_attestation",
        lambda **_kwargs: "a" * 64,
    )

    def fake_process(command, **kwargs):
        del kwargs
        process_commands.append(list(command))
        return 0, _structured_jsonl(), b""

    monkeypatch.setattr(T, "_run_codex_process", fake_process)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with T.stage_codex_launcher(
            str(wrapper),
            expected_launcher_digest=expected_digest) as staged:
        result = T.run_codex_text_structured(
            "fixture prompt",
            schema,
            executable=staged.executable,
            expected_launcher_digest=staged.launcher_digest,
        )
        assert result.payload == {"answer": "ok"}
        assert len(version_calls) == 3
        assert all(call == [staged.executable, "--version"]
                   for call in version_calls)
        assert len(process_commands) == 1
        assert process_commands[0][0] == staged.executable


def test_staged_launcher_mutation_is_detected_at_context_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    expected_digest = hashlib.sha256(native.read_bytes()).hexdigest()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess, "run", _fake_staged_version_run(calls))

    with pytest.raises(
        T.CodexProposerFailure,
        match="staged Codex launcher identity changed during use",
    ):
        with T.stage_codex_launcher(
                str(wrapper),
                expected_launcher_digest=expected_digest) as staged:
            staged_path = Path(staged.executable)
            changed = staged_path.read_bytes()[:-2] + b"v2"
            staged_path.chmod(0o700)
            staged_path.write_bytes(changed)
            staged_path.chmod(0o500)
    assert calls and calls[0][0] != str(native)


def test_non_codex_version_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = tmp_path / "opaque-native"
    opaque.write_bytes(b"opaque native fixture")
    opaque.chmod(0o700)
    calls: list[list[str]] = []

    def git_version_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        kwargs["stdout"].write(b"git version 2.50.1\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(T.subprocess, "run", git_version_run)
    with pytest.raises(
        T.CodexProposerFailure,
        match="does not identify codex-cli",
    ):
        T.codex_cli_fingerprint(str(opaque))
    assert calls == [[str(opaque.resolve()), "--version"]]

    calls.clear()
    with pytest.raises(
        T.CodexProposerFailure,
        match="does not identify codex-cli",
    ):
        with T.stage_codex_launcher(
                str(opaque),
                expected_launcher_digest=hashlib.sha256(
                    opaque.read_bytes()).hexdigest()):
            raise AssertionError("non-Codex version output must not be yielded")
    assert len(calls) == 1
    staged_attempt = Path(calls[0][0])
    assert staged_attempt != opaque
    assert staged_attempt.parent.name.startswith("bongard-codex-launcher-")
    assert not staged_attempt.exists()


def test_structured_turn_runs_native_and_rejects_post_launch_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, native = _official_wrapper_tree(tmp_path)
    wrapper_before = wrapper.read_bytes()
    expected_native_digest = hashlib.sha256(native.read_bytes()).hexdigest()
    source_home = tmp_path / "source-codex-home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    version_calls: list[list[str]] = []
    process_commands: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess,
        "run",
        _fake_version_run(
            native,
            version_calls,
            output=(T.PINNED_CODEX_CLI_VERSION + "\n").encode("utf-8"),
        ),
    )
    monkeypatch.setattr(
        T,
        "_resolve_no_tools_attestation",
        lambda **_kwargs: "a" * 64,
    )

    def fake_process(command, **kwargs):
        del kwargs
        command = list(command)
        process_commands.append(command)
        old = native.read_bytes()
        native.write_bytes(old[:-2] + b"v2")
        return 0, b"", b""

    monkeypatch.setattr(T, "_run_codex_process", fake_process)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with pytest.raises(
        T.CodexProposerFailure,
        match="launcher changed during text execution",
    ):
        T.run_codex_text_structured(
            "fixture prompt",
            schema,
            executable=str(wrapper),
            expected_launcher_digest=expected_native_digest,
        )

    assert version_calls == [[str(native), "--version"]]
    assert len(process_commands) == 1
    assert process_commands[0][0] == str(native)
    assert wrapper.read_bytes() == wrapper_before


def test_wrapper_digest_is_not_an_accepted_external_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _native = _official_wrapper_tree(tmp_path)
    calls: list[object] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("wrapper digest mismatch must precede execution")

    monkeypatch.setattr(T.subprocess, "run", forbidden)
    with pytest.raises(T.CodexProposerFailure, match="external commitment"):
        T.codex_cli_authenticated_fingerprint(
            str(wrapper),
            expected_launcher_digest=hashlib.sha256(
                wrapper.read_bytes()).hexdigest(),
        )
    assert calls == []


def test_direct_flat_fake_executable_remains_flat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flat = tmp_path / "flat-fake-codex"
    flat.write_bytes(b"offline flat executable fixture")
    flat.chmod(0o700)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        T.subprocess, "run", _fake_version_run(flat.resolve(), calls))

    fingerprint = T.codex_cli_fingerprint(str(flat))

    assert calls == [[str(flat.resolve()), "--version"]]
    assert fingerprint["launcher_digest"] == hashlib.sha256(
        flat.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "AMD64", (
            "x86_64-unknown-linux-musl", "@openai/codex-linux-x64",
            "linux", "x64")),
        ("Linux", "arm64", (
            "aarch64-unknown-linux-musl", "@openai/codex-linux-arm64",
            "linux", "arm64")),
        ("Darwin", "x86_64", (
            "x86_64-apple-darwin", "@openai/codex-darwin-x64",
            "darwin", "x64")),
        ("Darwin", "aarch64", (
            "aarch64-apple-darwin", "@openai/codex-darwin-arm64",
            "darwin", "arm64")),
        ("Windows", "AMD64", (
            "x86_64-pc-windows-msvc", "@openai/codex-win32-x64",
            "win32", "x64")),
        ("win32", "ARM64", (
            "aarch64-pc-windows-msvc", "@openai/codex-win32-arm64",
            "win32", "arm64")),
    ],
)
def test_cross_platform_official_package_mapping(
    system: str,
    machine: str,
    expected: tuple[str, str, str, str],
) -> None:
    assert T._codex_platform_package(system, machine) == expected


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("unknown.js", b"#!/usr/bin/env node\nconsole.log('shim')\n"),
        ("codex", b"#!/bin/sh\nexec /mutable/codex \"$@\"\n"),
    ],
)
def test_unrecognized_script_launchers_fail_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    contents: bytes,
) -> None:
    launcher = tmp_path / name
    launcher.write_bytes(contents)
    launcher.chmod(0o700)
    calls: list[object] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unrecognized scripts must not execute")

    monkeypatch.setattr(T.subprocess, "run", forbidden)
    with pytest.raises(
        T.CodexProposerFailure,
        match="unrecognized script/interpreter launcher",
    ):
        T.codex_cli_fingerprint(str(launcher))
    assert calls == []
