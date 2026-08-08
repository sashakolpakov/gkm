from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_contiguous_watchdog as W


def _config(tmp_path: Path) -> SimpleNamespace:
    runtime = tmp_path / "snapshot"
    runtime.mkdir()
    config_path = tmp_path / "operator.json"
    config_path.write_text("{}\n", encoding="ascii")
    config_path.chmod(0o400)
    return SimpleNamespace(
        campaign_root=(tmp_path / "campaign").resolve(),
        config_sha256="a" * 64,
        runtime_control_snapshot_root=runtime.resolve(),
        python_executable=Path(sys.executable).resolve(),
        config_path=config_path.resolve(),
    )


def _preflight_config(
    tmp_path: Path, *, selective: bool
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _config(tmp_path)
    config.terminal_condition = (
        W.Orchestrator.SELECTIVE_TERMINAL_CONDITION
        if selective
        else W.Orchestrator.CANONICAL_TERMINAL_CONDITION
    )
    config.frontier_import_root = (
        (tmp_path / "frontier-authority").resolve()
        if selective
        else None
    )
    config.selective_continuation_game = (
        "lf52" if selective else None
    )
    config.selective_frontier_import_sha256 = (
        "f" * 64 if selective else None
    )
    config.launch_attestation = tmp_path / "prelaunch.json"
    config.conformance_result = config.launch_attestation
    config.canonical_root = tmp_path / "canonical"
    config.environments_root = tmp_path / "environments"
    config.python_executable_sha256 = "b" * 64
    config.python_runtime_manifest = tmp_path / "runtime.json"
    config.python_runtime_manifest_sha256 = "c" * 64
    config.pilot_gate_receipt = tmp_path / "pilot.json"
    config.pilot_authentication_key = tmp_path / "pilot.key"
    config.pilot_production_stack_attestation_sha256 = "d" * 64
    config.backend_configuration = SimpleNamespace(
        image_digest="sha256:" + "e" * 64
    )
    config.auxiliary_backend_configuration = object()
    config.auxiliary_launch_configuration = object()
    return config


def _full_preflight_authority() -> dict[str, object]:
    body = {
        "kind": "arc_agi3_contiguous_receipt_launch_authority",
        "games": W.Supervisor.EXPECTED_GAMES,
        "levels": W.Supervisor.EXPECTED_LEVELS,
    }
    authority_sha256 = W.Orchestrator._json_sha256(body)
    evidence = {**body, "authority_sha256": authority_sha256}
    return {
        "status": "PASS",
        "launch_authority": "RECEIPT_DERIVED",
        "launch_authority_sha256": authority_sha256,
        "launch_authority_evidence": evidence,
        "games": W.Supervisor.EXPECTED_GAMES,
        "levels": W.Supervisor.EXPECTED_LEVELS,
    }


def _selective_preflight_authority(
    config: SimpleNamespace,
) -> dict[str, object]:
    control_body = {
        "kind":
            "arc_agi3_selective_continuation_control_authority",
        "terminal_release_authority": False,
    }
    control_sha256 = W.Orchestrator._json_sha256(control_body)
    control = {
        **control_body,
        "authority_sha256": control_sha256,
    }
    body = {
        "kind": "arc_agi3_selective_frontier_launch_authority",
        "operator_configuration_sha256": config.config_sha256,
        "frontier_import_root": str(config.frontier_import_root),
        "selective_continuation_game":
            config.selective_continuation_game,
        "selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "operator_authorized_selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "control_launch_authority_sha256": control_sha256,
        "control_launch_authority_kind": control["kind"],
        "control_launch_authority_evidence": control,
    }
    authority_sha256 = W.Orchestrator._json_sha256(body)
    evidence = {**body, "authority_sha256": authority_sha256}
    return {
        "status": "PASS",
        "launch_authority":
            "SELECTIVE_FRONTIER_RECEIPT_DERIVED",
        "launch_authority_kind":
            "arc_agi3_selective_frontier_launch_authority",
        "launch_authority_sha256": authority_sha256,
        "launch_authority_evidence": evidence,
        "selective_continuation_game":
            config.selective_continuation_game,
        "selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "operator_authorized_selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "image_digest": config.backend_configuration.image_digest,
    }


def _terminal(status: str) -> dict[str, object]:
    body = {
        "schema": 1,
        "kind": "test_operator_terminal",
        "status": status,
        "reason_code": "test_terminal",
        "error_class": "TestError",
    }
    return {
        **body,
        "receipt_sha256": W.Orchestrator._json_sha256(body),
    }


def _write_fake_operator(
    path: Path,
    *,
    terminal_after: int | None,
    sleep_seconds: float = 0.15,
) -> None:
    counter = path.with_suffix(".counter")
    terminal = _terminal("PREFLIGHT_FAILED")
    path.write_text(
        "\n".join([
            "import json",
            "import os",
            "import pathlib",
            "import sys",
            "import time",
            f"counter = pathlib.Path({str(counter)!r})",
            "value = int(counter.read_text()) + 1 if counter.exists() else 1",
            "counter.write_text(str(value))",
            f"time.sleep({sleep_seconds!r})",
            *(
                [
                    f"if value >= {terminal_after}:",
                    "    raw = json.dumps("
                    f"{terminal!r}, sort_keys=True, separators=(',', ':'))",
                    "    sys.stdout.write(raw + '\\n')",
                    "    sys.stdout.flush()",
                    "    raise SystemExit(2)",
                ]
                if terminal_after is not None
                else []
            ),
            "os._exit(17)",
        ])
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o500)


def _patch_lightweight_policy(
    monkeypatch, *, max_restarts: int
) -> None:
    monkeypatch.setattr(W, "_production_preflight", lambda _config: {})
    monkeypatch.setattr(W, "WATCHDOG_MAX_RESTARTS", max_restarts)
    monkeypatch.setattr(
        W,
        "WATCHDOG_RESTART_BACKOFF_SECONDS",
        tuple(0.0 for _ in range(max_restarts)),
    )
    monkeypatch.setattr(W, "WATCHDOG_POLL_SECONDS", 0.01)


def test_watchdog_full_preflight_requires_terminal_183_authority(
    tmp_path, monkeypatch
):
    config = _preflight_config(tmp_path, selective=False)
    observed = {}

    def launch(attestation, **kwargs):
        observed["attestation"] = attestation
        observed["kwargs"] = kwargs
        return _full_preflight_authority()

    monkeypatch.setattr(W.Supervisor, "launch_preflight", launch)
    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda _config: pytest.fail(
            "full mode must not request selective authority"
        ),
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_verify_auxiliary_backend_configuration",
        lambda backend, auxiliary: observed.update({
            "auxiliary": (backend, auxiliary)
        }),
    )

    result = W._production_preflight(config)
    assert result["launch_authority"] == "RECEIPT_DERIVED"
    assert observed["attestation"] == config.launch_attestation
    assert observed["kwargs"]["canonical_root"] == (
        config.canonical_root
    )
    assert observed["auxiliary"] == (
        config.auxiliary_backend_configuration,
        config.auxiliary_launch_configuration,
    )


def test_watchdog_selective_preflight_uses_nonterminal_frontier_authority(
    tmp_path, monkeypatch
):
    config = _preflight_config(tmp_path, selective=True)
    observed = {}

    monkeypatch.setattr(
        W.Supervisor,
        "launch_preflight",
        lambda *_args, **_kwargs: pytest.fail(
            "selective mode must not request terminal 183 authority"
        ),
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda selected: (
            observed.setdefault("config", selected)
            and _selective_preflight_authority(selected)
        ),
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_verify_auxiliary_backend_configuration",
        lambda backend, auxiliary: observed.update({
            "auxiliary": (backend, auxiliary)
        }),
    )

    result = W._production_preflight(config)
    assert result["launch_authority"] == (
        "SELECTIVE_FRONTIER_RECEIPT_DERIVED"
    )
    assert result["launch_authority_evidence"][
        "control_launch_authority_evidence"
    ]["terminal_release_authority"] is False
    assert observed["config"] is config
    assert observed["auxiliary"] == (
        config.auxiliary_backend_configuration,
        config.auxiliary_launch_configuration,
    )


def test_watchdog_campaign_modes_reject_cross_authority(
    tmp_path, monkeypatch
):
    full = _preflight_config(tmp_path / "full", selective=False)
    selective = _preflight_config(
        tmp_path / "selective", selective=True
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_verify_auxiliary_backend_configuration",
        lambda *_args: pytest.fail(
            "cross-mode authority must fail before auxiliary admission"
        ),
    )
    monkeypatch.setattr(
        W.Supervisor,
        "launch_preflight",
        lambda *_args, **_kwargs:
            _selective_preflight_authority(selective),
    )
    with pytest.raises(
        W.WatchdogError, match="another campaign mode"
    ):
        W._production_preflight(full)

    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda _config: _full_preflight_authority(),
    )
    with pytest.raises(
        W.WatchdogError, match="another campaign mode"
    ):
        W._production_preflight(selective)


def test_watchdog_selective_preflight_rejects_wrong_frontier_digest(
    tmp_path, monkeypatch
):
    config = _preflight_config(tmp_path, selective=True)
    authority = _selective_preflight_authority(config)
    authority["selective_frontier_import_sha256"] = "0" * 64
    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda _config: authority,
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_verify_auxiliary_backend_configuration",
        lambda *_args: pytest.fail(
            "wrong frontier digest reached auxiliary admission"
        ),
    )
    with pytest.raises(
        W.WatchdogError, match="another campaign mode"
    ):
        W._production_preflight(config)


def test_watchdog_recomputes_outer_and_nested_authority_hashes(
    tmp_path, monkeypatch
):
    full = _preflight_config(tmp_path / "full", selective=False)
    forged_full = _full_preflight_authority()
    forged_full["launch_authority_sha256"] = "f" * 64
    forged_full["launch_authority_evidence"][
        "authority_sha256"
    ] = "f" * 64
    monkeypatch.setattr(
        W.Supervisor,
        "launch_preflight",
        lambda *_args, **_kwargs: forged_full,
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_verify_auxiliary_backend_configuration",
        lambda *_args: pytest.fail(
            "forged outer authority reached auxiliary admission"
        ),
    )
    with pytest.raises(
        W.WatchdogError, match="another campaign mode"
    ):
        W._production_preflight(full)

    selective = _preflight_config(
        tmp_path / "selective", selective=True
    )
    forged_selective = _selective_preflight_authority(selective)
    outer = forged_selective["launch_authority_evidence"]
    control = outer["control_launch_authority_evidence"]
    control["authority_sha256"] = "f" * 64
    outer["control_launch_authority_sha256"] = "f" * 64
    outer_body = {
        key: value
        for key, value in outer.items()
        if key != "authority_sha256"
    }
    outer_sha256 = W.Orchestrator._json_sha256(outer_body)
    outer["authority_sha256"] = outer_sha256
    forged_selective["launch_authority_sha256"] = outer_sha256
    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda _config: forged_selective,
    )
    with pytest.raises(
        W.WatchdogError, match="another campaign mode"
    ):
        W._production_preflight(selective)


@pytest.mark.parametrize(
    ("frontier", "game", "terminal"),
    (
        (Path("/authority"), None, "selective_complete"),
        (None, "lf52", "selective_complete"),
        (Path("/authority"), "lf52", "selective_complete"),
        (Path("/authority"), "lf52", "complete"),
        (None, None, "selective_complete"),
    ),
)
def test_watchdog_rejects_inexact_campaign_mode_before_preflight(
    tmp_path, monkeypatch, frontier, game, terminal
):
    config = _preflight_config(tmp_path, selective=False)
    config.frontier_import_root = frontier
    config.selective_continuation_game = game
    config.terminal_condition = terminal
    monkeypatch.setattr(
        W.Supervisor,
        "launch_preflight",
        lambda *_args, **_kwargs: pytest.fail(
            "inexact mode reached terminal authority"
        ),
    )
    monkeypatch.setattr(
        W.Orchestrator,
        "_selective_operator_preflight",
        lambda _config: pytest.fail(
            "inexact mode reached selective authority"
        ),
    )
    with pytest.raises(
        W.WatchdogError, match="exact operator configuration"
    ):
        W._production_preflight(config)


def test_watchdog_restarts_crashed_operator_and_returns_exact_terminal(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    script = (
        config.runtime_control_snapshot_root
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_orchestrator.py"
    )
    script.parent.mkdir(parents=True)
    _write_fake_operator(script, terminal_after=2)
    _patch_lightweight_policy(
        monkeypatch, max_restarts=3
    )

    result = W.run_watchdog(config)
    assert result["status"] == "PREFLIGHT_FAILED"
    root = config.campaign_root / W.WATCHDOG_ROOT_NAME
    invocations = sorted(root.glob("invocation_*"))
    assert [path.name for path in invocations] == [
        "invocation_00",
        "invocation_01",
    ]
    for invocation in invocations:
        for name in ("stdout.json", "stderr.bin"):
            metadata = (invocation / name).stat(
                follow_symlinks=False
            )
            assert stat.S_IMODE(metadata.st_mode) == 0o400
    recovery = json.loads(
        (invocations[0] / "recovery.json").read_text(
            encoding="ascii"
        )
    )
    assert recovery["status"] == "NONTERMINAL_SEALED"
    assert recovery["reason_code"] == (
        "operator_exit_without_terminal_authority"
    )
    assert not (invocations[1] / "recovery.json").exists()


def test_watchdog_restart_exhaustion_is_durable_human_intervention(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    script = (
        config.runtime_control_snapshot_root
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_orchestrator.py"
    )
    script.parent.mkdir(parents=True)
    _write_fake_operator(script, terminal_after=None)
    _patch_lightweight_policy(
        monkeypatch, max_restarts=2
    )

    result = W.run_watchdog(config)
    assert result["status"] == "OPERATOR_INCIDENT"
    assert result["reason_code"] == "watchdog_restart_exhausted"
    assert result["runner_incident"][
        "human_intervention_required"
    ] is True
    incident = config.campaign_root / "operator_incident.json"
    assert json.loads(incident.read_text(encoding="ascii")) == result
    for invocation in (
        config.campaign_root / W.WATCHDOG_ROOT_NAME
    ).glob("invocation_*"):
        recovery = json.loads(
            (invocation / "recovery.json").read_text(
                encoding="ascii"
            )
        )
        assert recovery["status"] == "NONTERMINAL_SEALED"


def test_watchdog_recovers_terminal_written_before_service_crash(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    monkeypatch.setattr(W, "_production_preflight", lambda _config: {})
    monkeypatch.setattr(
        W,
        "_operator_command",
        lambda _config: pytest.fail("recovered terminal must not relaunch"),
    )
    W.Orchestrator._ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    root = config.campaign_root / W.WATCHDOG_ROOT_NAME
    W.Supervisor._operator_lease_private_directory(
        root,
        create=True,
        label="operator watchdog root",
    )
    invocation = root / "invocation_00"
    W.Supervisor._operator_lease_private_directory(
        invocation,
        create=True,
        label="watchdog invocation root",
    )
    terminal = _terminal("PREFLIGHT_FAILED")
    (invocation / "stdout.json").write_bytes(
        W._canonical_json(terminal) + b"\n"
    )
    (invocation / "stderr.bin").write_bytes(b"")
    (invocation / "stdout.json").chmod(0o400)
    (invocation / "stderr.bin").chmod(0o400)

    assert W.run_watchdog(config) == terminal


def test_watchdog_rejects_stdout_only_pass_and_duplicate_owner(
    tmp_path,
):
    config = _config(tmp_path)
    forged = _terminal("PASS")
    with pytest.raises(
        W.WatchdogError, match="no reopened terminal authority"
    ):
        W._validate_terminal_authority(
            config, forged, returncode=0
        )

    W.Orchestrator._ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    with W.WatchdogLease(config.campaign_root):
        with pytest.raises(
            W.WatchdogError,
            match="another live contiguous watchdog",
        ):
            with W.WatchdogLease(config.campaign_root):
                pass


def test_watchdog_accepts_only_durable_storage_terminal(tmp_path):
    config = _config(tmp_path)
    W.Orchestrator._ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    terminal = _terminal("JOURNAL_OR_STORAGE_EXHAUSTED")
    storage_path = (
        config.campaign_root / "operator_storage_exhausted.json"
    )
    storage_path.write_bytes(
        W._canonical_json(terminal) + b"\n"
    )
    storage_path.chmod(0o400)
    assert W._validate_terminal_authority(
        config, terminal, returncode=2
    ) == terminal
    with pytest.raises(W.WatchdogError):
        W._validate_terminal_authority(
            config, terminal, returncode=0
        )


def test_watchdog_command_is_exact_sealed_operator_entry(tmp_path):
    config = _config(tmp_path)
    operator = (
        config.runtime_control_snapshot_root
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_orchestrator.py"
    )
    operator.parent.mkdir(parents=True)
    operator.write_text("# sealed test operator\n", encoding="ascii")
    command = W._operator_command(config)
    assert command == (
        str(config.python_executable),
        "-I",
        "-E",
        "-s",
        "-B",
        str(operator),
        "--config",
        str(config.config_path),
    )
    W.Orchestrator._ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    W.Supervisor._operator_lease_private_directory(
        config.campaign_root / W.WATCHDOG_ROOT_NAME,
        create=True,
        label="operator watchdog root",
    )
    assert "PYTHONPATH" not in W._operator_environment(
        config.campaign_root / W.WATCHDOG_ROOT_NAME
    )


def test_watchdog_does_not_reapply_startup_deadline_after_active_lease(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    identity = "process-start"
    clock = {"seconds": 0.0, "sleeps": 0, "observations": 0}

    class Process:
        pid = 4242

        def poll(self):
            return 0 if clock["sleeps"] >= 3 else None

        def wait(self):
            return 0

    def observe(*_args, **_kwargs):
        clock["observations"] += 1
        if clock["observations"] == 2:
            raise W.Supervisor.SupervisorContractError(
                "transient authenticated-read interruption"
            )
        return {
            "heartbeat": {
                "status": (
                    "ACTIVE"
                    if clock["observations"] == 1
                    else "RELEASED"
                ),
                "owner_pid": Process.pid,
                "owner_process_start_identity_sha256": identity,
                "heartbeat_at_ns": 100_000_000_000,
            },
        }

    def sleep(_seconds):
        clock["sleeps"] += 1
        clock["seconds"] += 20.0

    monkeypatch.setattr(
        W.Supervisor.OperatorLease, "observe_current", observe
    )
    monkeypatch.setattr(
        W,
        "_terminate_exact_process",
        lambda *_args, **_kwargs: pytest.fail(
            "active/released operator was killed by startup timeout"
        ),
    )
    monkeypatch.setattr(
        W, "WATCHDOG_STARTUP_TIMEOUT_SECONDS", 1.0
    )
    assert W._monitor_started_operator(
        config,
        Process(),
        started_identity=identity,
        sleeper=sleep,
        monotonic=lambda: clock["seconds"],
        wall_time_ns=lambda: (
            100_000_000_000
            + int(clock["seconds"] * 1_000_000_000)
        ),
    ) == 0
    assert clock["observations"] == 3


def test_watchdog_real_child_survives_expired_startup_and_delayed_terminal(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    script = (
        config.runtime_control_snapshot_root
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_orchestrator.py"
    )
    script.parent.mkdir(parents=True)
    _write_fake_operator(
        script, terminal_after=1, sleep_seconds=0.35
    )
    _patch_lightweight_policy(monkeypatch, max_restarts=2)
    monkeypatch.setattr(
        W, "WATCHDOG_STARTUP_TIMEOUT_SECONDS", 0.02
    )
    monkeypatch.setattr(
        W, "WATCHDOG_HEARTBEAT_STALE_SECONDS", 0.5
    )
    monkeypatch.setattr(
        W, "WATCHDOG_RELEASE_EXIT_GRACE_SECONDS", 0.5
    )
    launched = {}

    def factory(*args, **kwargs):
        process = W.subprocess.Popen(*args, **kwargs)
        launched["process"] = process
        launched["started"] = W.time.monotonic()
        launched["identity"] = W._exact_process_identity(
            process.pid
        )
        return process

    def observe(*_args, **_kwargs):
        process = launched["process"]
        elapsed = W.time.monotonic() - launched["started"]
        if 0.10 <= elapsed < 0.18:
            raise W.Supervisor.SupervisorContractError(
                "injected lease reconciliation read"
            )
        return {
            "heartbeat": {
                "status": (
                    "RELEASED" if elapsed >= 0.18 else "ACTIVE"
                ),
                "owner_pid": process.pid,
                "owner_process_start_identity_sha256":
                    launched["identity"],
                "heartbeat_at_ns": W.time.time_ns(),
            },
        }

    monkeypatch.setattr(
        W.Supervisor.OperatorLease, "observe_current", observe
    )
    result = W.run_watchdog(
        config, process_factory=factory
    )
    assert result["status"] == "PREFLIGHT_FAILED"
    assert launched["process"].returncode == 2
    assert (
        config.campaign_root
        / W.WATCHDOG_ROOT_NAME
        / "invocation_00"
        / "stdout.json"
    ).read_bytes() == W._canonical_json(result) + b"\n"
