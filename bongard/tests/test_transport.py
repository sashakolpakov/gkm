"""Offline tests for the canonical headless Codex transport.

The subprocess boundary is faked throughout.  These tests never invoke Codex
or make a paid/model request.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Callable

import numpy as np
from PIL import Image
import pytest

import bongard.transport as T


THREAD_ID = "12345678-1234-4abc-9234-abcdef123456"
MODEL = "gpt-5.6-sol"
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def source_codex_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Never let an offline transport test read the developer's Codex home."""

    source = tmp_path / "source-codex-home"
    source.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source))
    return source


def _policy_cache_bytes() -> bytes:
    # Deliberately non-canonical whitespace: the exact source bytes, not a
    # decoded/re-encoded surrogate, must be copied and receipt-bound.
    return (
        b'{ "signed_payload": {"opaque_future_field": [1, true, null]}, '
        b'"signature": "test-signature" }\n'
    )


def _policy_cache_bytes_with_marker(marker: str) -> bytes:
    return json.dumps(
        {
            "signed_payload": {
                "bundle": {"config_toml": marker},
                "cached_at": f"2026-08-05T23:{marker}:00Z",
                "expires_at": f"2026-08-06T00:{marker}:00Z",
            },
            "signature": f"test-signature-{marker}",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_panel(
    path: Path,
    marker: int,
    *,
    mode: str = "RGB",
    antialias: bool = True,
) -> bytes:
    gray = np.full((18, 19), 255, dtype=np.uint8)
    gray[2 + marker % 12, 2:16] = 0
    if antialias:
        gray[3 + marker % 12, 2:16] = 127
        gray[4 + marker % 12, 2:16] = 128
    if mode == "RGB":
        raster = np.repeat(gray[..., None], 3, axis=2)
    else:
        raster = gray
    Image.fromarray(raster, mode=mode).save(path, format="PNG")
    return path.read_bytes()


@pytest.fixture
def panel_paths(tmp_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            _write_panel(path, index + (0 if side == "pos" else 6))
            paths.append(str(path.resolve()))
    return tuple(paths)


def _jsonl(payload: Any = None, *, model: str | None = None) -> bytes:
    payload = {"answer": "ok"} if payload is None else payload
    started: dict[str, Any] = {
        "type": "thread.started",
        "thread_id": THREAD_ID,
    }
    if model is not None:
        started["model"] = model
    events = [
        started,
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 17,
                "cached_input_tokens": 3,
                "output_tokens": 5,
                "reasoning_output_tokens": 2,
            },
        },
    ]
    return (
        "\n".join(json.dumps(event, separators=(",", ":")) for event in events)
        + "\n"
    ).encode("utf-8")


class _CallLog(list[tuple[list[str], dict[str, Any]]]):
    processes: list[Any]


def _install_fake_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: bytes,
    inspect_exec: Callable[[list[str], dict[str, Any]], None] | None = None,
) -> _CallLog:
    calls = _CallLog()
    calls.processes = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if len(command) == 2 and command[1] == "--version":
            kwargs["stdout"].write(b"codex-cli 0.test\n")
            kwargs["stdout"].flush()
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"unexpected subprocess.run call: {command!r}")

    class FakeProcess:
        def __init__(self, command: list[str], kwargs: dict[str, Any]):
            self.command = command
            self.kwargs = kwargs
            self.pid = 90_000_000 + len(calls.processes)
            self.returncode: int | None = None

        def communicate(self, *, input: bytes, timeout: int):
            assert input
            assert timeout > 0
            self.kwargs["stdout"].write(stdout)
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
        calls.append((command, kwargs))
        assert "exec" in command
        if inspect_exec is not None:
            inspect_exec(command, kwargs)
        process = FakeProcess(command, kwargs)
        calls.processes.append(process)
        return process

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)
    return calls


def _fake_launcher(tmp_path: Path) -> str:
    launcher = tmp_path / "fake-codex"
    launcher.write_bytes(b"#!/bin/sh\nexit 99\n")
    launcher.chmod(0o700)
    return str(launcher.resolve())


def test_authenticated_fingerprint_rejects_bytes_before_version_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _fake_launcher(tmp_path)
    calls: list[object] = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("mismatched launcher must not execute")

    monkeypatch.setattr(T.subprocess, "run", forbidden_run)
    with pytest.raises(
        T.CodexProposerFailure,
        match="external commitment",
    ):
        T.codex_cli_authenticated_fingerprint(
            launcher,
            expected_launcher_digest="0" * 64,
        )
    assert calls == []


def test_structured_turn_rejects_launcher_before_version_execution(
    panel_paths: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _fake_launcher(tmp_path)
    calls: list[object] = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("mismatched launcher must not execute")

    monkeypatch.setattr(T.subprocess, "run", forbidden_run)
    with pytest.raises(
        T.CodexProposerFailure,
        match="external commitment",
    ):
        T.run_codex_structured(
            "task",
            panel_paths,
            SIMPLE_SCHEMA,
            executable=launcher,
            expected_launcher_digest="0" * 64,
        )
    assert calls == []


def test_named_transport_rejects_unsupported_schema_before_launcher(
    panel_paths: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _fake_launcher(tmp_path)
    calls: list[object] = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid schema must fail before launcher execution")

    monkeypatch.setattr(T.subprocess, "run", forbidden_run)
    invalid = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    with pytest.raises(
        T.CodexProposerFailure,
        match="unsupported keywords: minItems",
    ):
        T.run_codex_named_images_structured(
            "task",
            (panel_paths[0],),
            ("query.png",),
            invalid,
            executable=launcher,
        )
    assert calls == []


def test_structured_turn_copies_exact_rgb_bytes_without_repository_exposure(
    panel_paths: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {Path(path).name: Path(path).read_bytes() for path in panel_paths}
    observed: dict[str, Any] = {}
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "invocation-only-test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")

    def inspect_exec(command: list[str], kwargs: dict[str, Any]) -> None:
        view = Path(kwargs["cwd"]).resolve()
        observed["view"] = view
        observed["command"] = command
        observed["env"] = dict(kwargs["env"])
        observed["mode"] = stat.S_IMODE(view.stat().st_mode)
        observed["files"] = sorted(path.name for path in view.iterdir())
        observed["images"] = {
            name: (view / name).read_bytes() for name in T._PANEL_NAMES
        }
        observed["schema"] = (view / "output_schema.json").read_bytes()
        observed["auth_home"] = Path(kwargs["env"]["CODEX_HOME"]).resolve()
        observed["auth_files"] = sorted(
            path.name for path in observed["auth_home"].iterdir()
        )

    calls = _install_fake_cli(
        monkeypatch,
        stdout=_jsonl(model=MODEL),
        inspect_exec=inspect_exec,
    )
    result = T.run_codex_structured(
        "solve only the attached support panels",
        panel_paths,
        SIMPLE_SCHEMA,
        model=MODEL,
        executable=launcher,
    )

    assert result.payload == {"answer": "ok"}
    assert observed["images"] == expected
    assert observed["schema"] == T._canonical_json_bytes(SIMPLE_SCHEMA)
    assert observed["mode"] == 0o700
    assert observed["files"] == sorted([*T._PANEL_NAMES, "output_schema.json"])
    assert os.path.commonpath((str(observed["view"]), T.WORKSPACE_ROOT)) != (
        T.WORKSPACE_ROOT
    )
    assert os.path.commonpath((str(observed["auth_home"]), T.WORKSPACE_ROOT)) != (
        T.WORKSPACE_ROOT
    )
    assert observed["auth_files"] == []
    command = observed["command"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--cd") + 1] == str(observed["view"])
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--strict-config" in command
    assert "--ask-for-approval" in command
    assert "never" in command
    disabled = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--disable"
    ]
    assert disabled == list(T._DISABLED_FEATURES)
    for secret in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "UNRELATED_SECRET"):
        assert secret not in observed["env"]
    assert len(calls) == 3  # version, turn, version
    assert result.receipt.input_digest_schema == T.STRUCTURED_INPUT_DIGEST_SCHEMA
    assert result.receipt.panel_view_digest == T.ordered_panel_view_digest(panel_paths)
    assert result.receipt.output_schema_digest == hashlib.sha256(
        observed["schema"]
    ).hexdigest()
    assert result.receipt.cloud_config_bundle_cache_binding == "absent"
    T.validate_codex_receipt(result.receipt.to_dict())


def test_signed_cloud_policy_cache_is_exactly_staged_and_receipt_bound(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _policy_cache_bytes()
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(cache)
    (source_codex_home / "config.toml").write_text("must not be copied")
    for directory in ("memories", "sessions", "plugins", "skills"):
        path = source_codex_home / directory
        path.mkdir()
        (path / "private-data").write_text("must not be copied")

    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    observed: dict[str, Any] = {}

    def inspect_exec(command: list[str], kwargs: dict[str, Any]) -> None:
        del command
        ephemeral = Path(kwargs["env"]["CODEX_HOME"])
        observed["files"] = sorted(path.name for path in ephemeral.iterdir())
        target = ephemeral / T._CLOUD_CONFIG_BUNDLE_CACHE
        observed["cache"] = target.read_bytes()
        observed["mode"] = stat.S_IMODE(target.stat().st_mode)

    _install_fake_cli(monkeypatch, stdout=_jsonl(), inspect_exec=inspect_exec)
    result = T.run_codex_structured(
        "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
    )

    assert observed == {
        "files": [T._CLOUD_CONFIG_BUNDLE_CACHE],
        "cache": cache,
        "mode": 0o600,
    }
    assert result.receipt.cloud_config_bundle_cache_binding == (
        "sha256:" + hashlib.sha256(cache).hexdigest()
    )
    T.validate_codex_receipt(result.receipt.to_dict())


def test_episode_policy_snapshot_survives_adversarial_live_cache_refresh(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _policy_cache_bytes_with_marker("11")
    refreshed = _policy_cache_bytes_with_marker("12")
    source = source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE
    source.write_bytes(first)
    snapshot = T.snapshot_cloud_policy_cache()
    assert snapshot.data == first

    # Simulate the global Codex process refreshing cached_at/expires_at and
    # the signature between episode calls.  Neither isolated call may consume
    # these later bytes once the episode snapshot exists.
    source.write_bytes(refreshed)
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    staged: list[bytes] = []

    def inspect_exec(command: list[str], kwargs: dict[str, Any]) -> None:
        del command
        target = Path(kwargs["env"]["CODEX_HOME"]) / T._CLOUD_CONFIG_BUNDLE_CACHE
        staged.append(target.read_bytes())

    _install_fake_cli(monkeypatch, stdout=_jsonl(), inspect_exec=inspect_exec)
    proposal = T.run_codex_structured(
        "task",
        panel_paths,
        SIMPLE_SCHEMA,
        executable=launcher,
        cloud_policy_cache_snapshot=snapshot,
    )
    observation = T.run_codex_named_images_structured(
        "task",
        (panel_paths[0],),
        ("query.png",),
        SIMPLE_SCHEMA,
        executable=launcher,
        cloud_policy_cache_snapshot=snapshot,
    )

    expected_binding = "sha256:" + hashlib.sha256(first).hexdigest()
    assert source.read_bytes() == refreshed
    assert staged == [first, first]
    assert proposal.receipt.cloud_config_bundle_cache_binding == expected_binding
    assert observation.receipt.cloud_config_bundle_cache_binding == expected_binding


def test_policy_snapshot_rejects_malformed_or_nonbyte_preimages() -> None:
    with pytest.raises(T.CodexProposerFailure, match="signed envelope"):
        T.CloudPolicyCacheSnapshot(b'{"not":"signed"}')
    with pytest.raises(T.CodexProposerFailure, match="exact bytes"):
        T.CloudPolicyCacheSnapshot("not-bytes")  # type: ignore[arg-type]


def test_cloud_policy_cache_source_symlink_is_rejected_without_launch(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside-policy-cache.json"
    outside.write_bytes(_policy_cache_bytes())
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).symlink_to(outside)
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    calls = _install_fake_cli(monkeypatch, stdout=_jsonl())

    with pytest.raises(T.CodexProposerFailure, match="singly-linked"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )
    assert not calls.processes


def test_oversized_cloud_policy_cache_is_rejected_without_launch(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(
        b"x" * (T.MAX_CLOUD_CONFIG_BUNDLE_CACHE_BYTES + 1)
    )
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    calls = _install_fake_cli(monkeypatch, stdout=_jsonl())

    with pytest.raises(T.CodexProposerFailure, match="bounded"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )
    assert not calls.processes


@pytest.mark.parametrize(
    "cache",
    [
        b'{}',
        b'{"signed_payload":{},"signature":"","extra":true}',
        b'{"signed_payload":"opaque","signature":"sig"}',
        b'{"signed_payload":{},"signature":{}}',
    ],
)
def test_cloud_policy_cache_requires_only_the_exact_signed_outer_envelope(
    cache: bytes,
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(cache)
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    calls = _install_fake_cli(monkeypatch, stdout=_jsonl())

    with pytest.raises(T.CodexProposerFailure, match="signed envelope"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )
    assert not calls.processes


def test_staged_cloud_policy_cache_mutation_is_detected(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(
        _policy_cache_bytes()
    )
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")

    def mutate_at_launch(command: list[str], kwargs: dict[str, Any]) -> None:
        del command
        target = Path(kwargs["env"]["CODEX_HOME"]) / T._CLOUD_CONFIG_BUNDLE_CACHE
        target.write_bytes(target.read_bytes() + b" ")

    _install_fake_cli(
        monkeypatch,
        stdout=_jsonl(),
        inspect_exec=mutate_at_launch,
    )
    with pytest.raises(T.CodexProposerFailure, match="metadata changed"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )


def test_prelaunch_policy_cache_mutation_is_detected_before_process_creation(
    panel_paths: tuple[str, ...],
    source_codex_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (source_codex_home / T._CLOUD_CONFIG_BUNDLE_CACHE).write_bytes(
        _policy_cache_bytes()
    )
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    original_stage = T._stage_cloud_policy_cache

    def stage_then_mutate(
        codex_home: str,
        snapshot: T.CloudPolicyCacheSnapshot | None = None,
    ):
        stage = original_stage(codex_home, snapshot)
        Path(stage.path).write_bytes(Path(stage.path).read_bytes() + b" ")
        return stage

    monkeypatch.setattr(T, "_stage_cloud_policy_cache", stage_then_mutate)
    calls = _install_fake_cli(monkeypatch, stdout=_jsonl())
    with pytest.raises(T.CodexProposerFailure, match="metadata changed"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )
    assert not calls.processes


def test_committed_policy_cache_absence_rejects_late_file_injection(
    panel_paths: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")

    def inject_at_launch(command: list[str], kwargs: dict[str, Any]) -> None:
        del command
        target = Path(kwargs["env"]["CODEX_HOME"]) / T._CLOUD_CONFIG_BUNDLE_CACHE
        target.write_bytes(_policy_cache_bytes())

    _install_fake_cli(
        monkeypatch,
        stdout=_jsonl(),
        inspect_exec=inject_at_launch,
    )
    with pytest.raises(T.CodexProposerFailure, match="appeared after absence"):
        T.run_codex_structured(
            "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
        )


def test_semantic_digest_thresholds_grayscale_rgb_but_raw_digest_does_not(
    tmp_path: Path,
) -> None:
    rgb_dir = tmp_path / "rgb"
    gray_dir = tmp_path / "gray"
    rgb_dir.mkdir()
    gray_dir.mkdir()
    rgb_paths: list[str] = []
    gray_paths: list[str] = []
    for side in ("pos", "neg"):
        for index in range(6):
            name = f"{side}_{index}.png"
            _write_panel(rgb_dir / name, index, mode="RGB", antialias=True)
            _write_panel(gray_dir / name, index, mode="L", antialias=True)
            rgb_paths.append(str((rgb_dir / name).resolve()))
            gray_paths.append(str((gray_dir / name).resolve()))

    semantic_digest = T.semantic_panel_set_digest(rgb_paths)
    assert semantic_digest == (
        "sha256:2fec28c0e0ccf0bbc2cb2f3cc2d594e39fc47cfd1ec5e7f8c9a09c4219103167"
    )
    assert semantic_digest == T.semantic_panel_set_digest(gray_paths)
    assert T.ordered_panel_view_digest(rgb_paths) != T.ordered_panel_view_digest(
        gray_paths
    )

    changed = np.full((18, 19, 3), 255, dtype=np.uint8)
    changed[2, 2:16] = 0
    changed[3, 2:16] = 128  # 128 is background under the strict <128 rule.
    Image.fromarray(changed, mode="RGB").save(rgb_dir / "pos_0.png", format="PNG")
    digest_at_128 = T.semantic_panel_set_digest(rgb_paths)
    changed[3, 2:16] = 127
    Image.fromarray(changed, mode="RGB").save(rgb_dir / "pos_0.png", format="PNG")
    assert T.semantic_panel_set_digest(rgb_paths) != digest_at_128


def test_named_turn_requires_neutral_names_and_binds_their_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = []
    for index in range(2):
        path = tmp_path / f"source-{index}.png"
        _write_panel(path, index)
        images.append(str(path.resolve()))
    names = ("image_a.png", "image_b.png")

    with pytest.raises(T.CodexProposerFailure, match="must not encode"):
        T.named_image_view_digest(images, ("pos_0.png", "image_b.png"))
    with pytest.raises(T.CodexProposerFailure, match="unique"):
        T.named_image_set_digest(images, ("same.png", "same.png"))

    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    observed: dict[str, Any] = {}

    def inspect_exec(command: list[str], kwargs: dict[str, Any]) -> None:
        view = Path(kwargs["cwd"])
        image_arguments = [
            Path(command[index + 1]).name
            for index, value in enumerate(command[:-1])
            if value == "--image"
        ]
        observed["names"] = tuple(image_arguments)
        observed["bytes"] = tuple((view / name).read_bytes() for name in names)

    _install_fake_cli(
        monkeypatch, stdout=_jsonl(), inspect_exec=inspect_exec
    )
    result = T.run_codex_named_images_structured(
        "score these opaque images",
        images,
        names,
        SIMPLE_SCHEMA,
        executable=launcher,
    )

    assert observed["names"] == names
    assert observed["bytes"] == tuple(Path(path).read_bytes() for path in images)
    assert result.receipt.input_digest_schema == T.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
    assert result.receipt.panel_view_digest == T.named_image_view_digest(images, names)
    assert result.receipt.panel_set_digest == T.named_image_set_digest(images, names)
    assert result.receipt.cloud_config_bundle_cache_binding == "absent"


def test_schema_and_receipt_are_strict_and_digest_bound(
    panel_paths: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _fake_launcher(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CODEX_API_KEY", "test-token")
    _install_fake_cli(monkeypatch, stdout=_jsonl())

    with pytest.raises(T.CodexProposerFailure, match="mapping"):
        T.run_codex_structured(
            "task", panel_paths, ["not", "a", "schema"], executable=launcher
        )
    with pytest.raises(T.CodexProposerFailure, match="finite"):
        T.run_codex_structured(
            "task",
            panel_paths,
            {"type": "object", "poison": float("nan")},
            executable=launcher,
        )

    result = T.run_codex_structured(
        "task", panel_paths, SIMPLE_SCHEMA, executable=launcher
    )
    receipt = result.receipt.to_dict()
    T.validate_codex_receipt(receipt)

    tampered = copy.deepcopy(receipt)
    tampered["panel_view_digest"] = "0" * 64
    with pytest.raises(T.CodexProposerFailure, match="does not reproduce"):
        T.validate_codex_receipt(tampered)

    cache_tampered = copy.deepcopy(receipt)
    cache_tampered["cloud_config_bundle_cache_binding"] = "sha256:" + "0" * 64
    with pytest.raises(T.CodexProposerFailure, match="does not reproduce"):
        T.validate_codex_receipt(cache_tampered)

    invalid_cache_binding = copy.deepcopy(receipt)
    invalid_cache_binding["cloud_config_bundle_cache_binding"] = "present"
    body = {
        key: value
        for key, value in invalid_cache_binding.items()
        if key != "receipt_digest"
    }
    invalid_cache_binding["receipt_digest"] = T._digest(body)
    with pytest.raises(T.CodexProposerFailure, match="cache binding is invalid"):
        T.validate_codex_receipt(invalid_cache_binding)

    extra = copy.deepcopy(receipt)
    extra["repository_path"] = T.WORKSPACE_ROOT
    with pytest.raises(T.CodexProposerFailure, match="fields"):
        T.validate_codex_receipt(extra)
