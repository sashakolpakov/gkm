#!/usr/bin/env python3
"""Independent fail-closed service for the contiguous campaign operator.

The operator owns all campaign authority.  This process owns none: it performs
the same immutable launch preflight, enforces one watchdog instance, starts the
exact sealed operator command, observes its authenticated lease heartbeat, and
restarts only after the prior PID/start identity is absent.  Exhaustion is a
durable operator incident with a fixed human-intervention request, never an
infinite crash loop or an interactive prompt.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _DIRECT_SCRIPT = Path(__file__).resolve()
    _DIRECT_CONTROL_ROOT = str(_DIRECT_SCRIPT.parent)
    _DIRECT_REPOSITORY_ROOT = str(_DIRECT_SCRIPT.parents[2])
    if _DIRECT_CONTROL_ROOT not in sys.path:
        sys.path.insert(0, _DIRECT_CONTROL_ROOT)
    if _DIRECT_REPOSITORY_ROOT not in sys.path:
        sys.path.insert(1, _DIRECT_REPOSITORY_ROOT)

import arc_agi3_contiguous_orchestrator as Orchestrator
import arc_agi3_contiguous_supervisor as Supervisor


WATCHDOG_SCHEMA = 1
WATCHDOG_ROOT_NAME = "operator_watchdog"
WATCHDOG_MAX_STREAM_BYTES = 4 * 1024 * 1024
WATCHDOG_MAX_RESTARTS = 6
WATCHDOG_RESTART_BACKOFF_SECONDS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
WATCHDOG_STARTUP_TIMEOUT_SECONDS = 15 * 60
WATCHDOG_HEARTBEAT_STALE_SECONDS = 45.0
WATCHDOG_POLL_SECONDS = 1.0
WATCHDOG_TERMINATION_GRACE_SECONDS = 10.0
WATCHDOG_RELEASE_EXIT_GRACE_SECONDS = 30.0
TERMINAL_STATUSES = frozenset({
    "PASS",
    "BLOCKED",
    "OPERATOR_INCIDENT",
    "PREFLIGHT_FAILED",
    "JOURNAL_OR_STORAGE_EXHAUSTED",
})


class WatchdogError(RuntimeError):
    """The independent operator service could not proceed safely."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise WatchdogError(
            "watchdog evidence is not canonical JSON"
        ) from exc


def _strict_terminal(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > WATCHDOG_MAX_STREAM_BYTES:
        raise WatchdogError(
            "operator terminal stream is empty or oversized"
        )

    def no_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise WatchdogError(
                    "operator terminal stream has duplicate keys"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except WatchdogError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WatchdogError(
            "operator terminal stream is not strict JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or _canonical_json(value) + b"\n" != raw
        or value.get("status") not in TERMINAL_STATUSES
    ):
        raise WatchdogError(
            "operator terminal stream is not a canonical terminal object"
        )
    return value


def _validate_terminal_authority(
    config: Orchestrator.OperatorConfiguration,
    value: Mapping[str, Any],
    *,
    returncode: int,
) -> dict[str, Any]:
    """Reopen durable authority; stdout alone can never end the service."""

    status = value.get("status")
    expected_returncode = (
        2
        if status
        in {
            "OPERATOR_INCIDENT",
            "PREFLIGHT_FAILED",
            "JOURNAL_OR_STORAGE_EXHAUSTED",
        }
        else 0
    )
    if returncode != expected_returncode:
        raise WatchdogError(
            "operator exit code differs from its terminal status"
        )
    selected = dict(value)
    terminal_path_value = selected.get("terminal_receipt")
    terminal_sha256 = selected.get("terminal_receipt_sha256")
    if terminal_path_value is not None or terminal_sha256 is not None:
        allowed = {
            config.campaign_root / "operator_incident.json",
            config.campaign_root / "operator_storage_exhausted.json",
            config.campaign_root / "operator_terminal_blocked.json",
            config.campaign_root / "terminal_audits" / "operator.json",
        }
        terminal_path = Path(str(terminal_path_value))
        if (
            terminal_path not in allowed
            or re.fullmatch(
                r"[0-9a-f]{64}", str(terminal_sha256)
            )
            is None
        ):
            raise WatchdogError(
                "operator terminal receipt path/hash is not canonical"
            )
        raw = Supervisor._read_regular_bytes(terminal_path)
        if (
            Orchestrator._sha256(raw) != terminal_sha256
            or len(raw) > Orchestrator.MAX_JSON_BYTES
        ):
            raise WatchdogError(
                "operator terminal receipt changed"
            )
        base = _strict_terminal(raw)
        if (
            base.get("status") != status
            or any(selected.get(key) != item for key, item in base.items())
        ):
            raise WatchdogError(
                "operator stdout differs from durable terminal receipt"
            )
        cleanup_path = Path(str(
            selected.get("terminal_cleanup_receipt", "")
        ))
        cleanup_sha256 = selected.get(
            "terminal_cleanup_receipt_sha256"
        )
        expected_cleanup = (
            config.campaign_root
            / "operator_canary_control"
            / "terminal_cleanup_receipt.json"
        )
        if (
            cleanup_path != expected_cleanup
            or re.fullmatch(
                r"[0-9a-f]{64}", str(cleanup_sha256)
            )
            is None
            or Orchestrator._sha256(
                Supervisor._read_regular_bytes(cleanup_path)
            )
            != cleanup_sha256
            or selected.get("canary_live_values_cleaned") is not True
        ):
            raise WatchdogError(
                "operator terminal lacks durable canary cleanup"
            )
        return selected
    receipt_sha256 = selected.get("receipt_sha256")
    if (
        status
        not in {
            "OPERATOR_INCIDENT",
            "PREFLIGHT_FAILED",
            "JOURNAL_OR_STORAGE_EXHAUSTED",
        }
        or re.fullmatch(
            r"[0-9a-f]{64}", str(receipt_sha256)
        )
        is None
        or receipt_sha256
        != Orchestrator._json_sha256({
            key: item
            for key, item in selected.items()
            if key != "receipt_sha256"
        })
    ):
        raise WatchdogError(
            "operator stdout has no reopened terminal authority"
        )
    if status in {
        "OPERATOR_INCIDENT",
        "JOURNAL_OR_STORAGE_EXHAUSTED",
    }:
        incident_path = config.campaign_root / (
            "operator_incident.json"
            if status == "OPERATOR_INCIDENT"
            else "operator_storage_exhausted.json"
        )
        if Supervisor._read_regular_bytes(incident_path) != (
            _canonical_json(selected) + b"\n"
        ):
            raise WatchdogError(
                "operator incident stdout is not durable"
            )
    return selected


def _limit_operator_child() -> None:
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (
            WATCHDOG_MAX_STREAM_BYTES,
            WATCHDOG_MAX_STREAM_BYTES,
        ),
    )


class WatchdogLease:
    """One kernel-held service owner; it never grants campaign authority."""

    def __init__(self, campaign_root: Path):
        self.root = Path(campaign_root) / WATCHDOG_ROOT_NAME
        self.path = self.root / "watchdog.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "WatchdogLease":
        Supervisor._operator_lease_private_directory(
            self.root.parent,
            create=True,
            label="watchdog campaign root",
        )
        Supervisor._operator_lease_private_directory(
            self.root,
            create=True,
            label="operator watchdog root",
        )
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise WatchdogError(
                "watchdog lock cannot be opened"
            ) from exc
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise WatchdogError(
                "watchdog lock is not an unaliased owner-held file"
            )
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            os.close(descriptor)
            raise WatchdogError(
                "another live contiguous watchdog owns this campaign"
            ) from exc
        self.descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        descriptor = self.descriptor
        self.descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _production_preflight(
    config: Orchestrator.OperatorConfiguration,
) -> Mapping[str, Any]:
    selective_game = getattr(
        config, "selective_continuation_game", None
    )
    frontier_import_root = getattr(
        config, "frontier_import_root", None
    )
    selective_frontier_import_sha256 = getattr(
        config, "selective_frontier_import_sha256", None
    )
    selective_mode = (
        selective_game is not None
        or frontier_import_root is not None
        or selective_frontier_import_sha256 is not None
    )
    expected_terminal = (
        Orchestrator.SELECTIVE_TERMINAL_CONDITION
        if selective_mode
        else Orchestrator.CANONICAL_TERMINAL_CONDITION
    )
    if (
        getattr(config, "terminal_condition", None)
        != expected_terminal
        or selective_mode
        and (
            selective_game is None
            or frontier_import_root is None
            or selective_frontier_import_sha256 is None
        )
    ):
        raise WatchdogError(
            "watchdog preflight mode differs from the exact operator "
            "configuration"
        )
    if selective_mode:
        result = Orchestrator._selective_operator_preflight(config)
    else:
        result = Supervisor.launch_preflight(
            config.launch_attestation,
            requested_image_digest=(
                config.backend_configuration.image_digest
            ),
            conformance_result=config.conformance_result,
            canonical_root=config.canonical_root,
            environments_root=config.environments_root,
            python_executable=config.python_executable,
            python_executable_sha256=config.python_executable_sha256,
            python_runtime_manifest=config.python_runtime_manifest,
            python_runtime_manifest_sha256=(
                config.python_runtime_manifest_sha256
            ),
            runtime_control_snapshot_root=(
                config.runtime_control_snapshot_root
            ),
            pilot_gate_receipt=config.pilot_gate_receipt,
            pilot_authentication_key=(
                config.pilot_authentication_key
            ),
            pilot_production_stack_attestation_sha256=(
                config.pilot_production_stack_attestation_sha256
            ),
        )
    if not isinstance(result, Mapping):
        raise WatchdogError(
            "watchdog preflight returned authority for another campaign "
            "mode"
        )
    evidence = result.get("launch_authority_evidence")
    authority_sha256 = result.get("launch_authority_sha256")

    def valid_authority_evidence(
        value: object, expected_sha256: object
    ) -> bool:
        if (
            not isinstance(value, Mapping)
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            is None
            or value.get("authority_sha256") != expected_sha256
        ):
            return False
        body = {
            key: item
            for key, item in value.items()
            if key != "authority_sha256"
        }
        try:
            return (
                Orchestrator._json_sha256(body)
                == expected_sha256
            )
        except (TypeError, ValueError):
            return False

    common_valid = (
        result.get("status") == "PASS"
        and valid_authority_evidence(evidence, authority_sha256)
    )
    if selective_mode:
        control = (
            evidence.get("control_launch_authority_evidence")
            if isinstance(evidence, Mapping)
            else None
        )
        authority_valid = (
            common_valid
            and result.get("launch_authority")
            == "SELECTIVE_FRONTIER_RECEIPT_DERIVED"
            and result.get("launch_authority_kind")
            == "arc_agi3_selective_frontier_launch_authority"
            and evidence.get("kind")
            == "arc_agi3_selective_frontier_launch_authority"
            and evidence.get("operator_configuration_sha256")
            == config.config_sha256
            and evidence.get("frontier_import_root")
            == str(frontier_import_root)
            and evidence.get("selective_continuation_game")
            == selective_game
            and result.get("selective_continuation_game")
            == selective_game
            and result.get("selective_frontier_import_sha256")
            == selective_frontier_import_sha256
            and result.get(
                "operator_authorized_selective_frontier_import_sha256"
            )
            == selective_frontier_import_sha256
            and evidence.get("selective_frontier_import_sha256")
            == selective_frontier_import_sha256
            and evidence.get(
                "operator_authorized_selective_frontier_import_sha256"
            )
            == selective_frontier_import_sha256
            and result.get("image_digest")
            == config.backend_configuration.image_digest
            and isinstance(control, Mapping)
            and control.get("kind")
            == "arc_agi3_selective_continuation_control_authority"
            and evidence.get("control_launch_authority_kind")
            == control.get("kind")
            and valid_authority_evidence(
                control,
                evidence.get("control_launch_authority_sha256"),
            )
            and control.get("terminal_release_authority") is False
        )
    else:
        authority_valid = (
            common_valid
            and result.get("launch_authority") == "RECEIPT_DERIVED"
            and evidence.get("kind")
            == "arc_agi3_contiguous_receipt_launch_authority"
            and result.get("games") == Supervisor.EXPECTED_GAMES
            and result.get("levels") == Supervisor.EXPECTED_LEVELS
            and evidence.get("games") == Supervisor.EXPECTED_GAMES
            and evidence.get("levels") == Supervisor.EXPECTED_LEVELS
        )
    if not authority_valid:
        raise WatchdogError(
            "watchdog preflight returned authority for another campaign "
            "mode"
        )
    Orchestrator._verify_auxiliary_backend_configuration(
        config.auxiliary_backend_configuration,
        config.auxiliary_launch_configuration,
    )
    return result


def _operator_command(
    config: Orchestrator.OperatorConfiguration,
) -> tuple[str, ...]:
    script = (
        config.runtime_control_snapshot_root
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_orchestrator.py"
    )
    try:
        metadata = script.stat(follow_symlinks=False)
    except OSError as exc:
        raise WatchdogError(
            "sealed operator script is unavailable"
        ) from exc
    if (
        script.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise WatchdogError(
            "sealed operator script is aliased or nonregular"
        )
    return (
        str(config.python_executable),
        "-I",
        "-E",
        "-s",
        "-B",
        str(script),
        "--config",
        str(config.config_path),
    )


def _operator_environment(root: Path) -> dict[str, str]:
    runtime = root / "runtime"
    Supervisor._operator_lease_private_directory(
        runtime,
        create=True,
        label="watchdog operator runtime",
    )
    return {
        "HOME": str(runtime),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": str(runtime),
    }


def _exact_process_identity(pid: int) -> str | None:
    try:
        return Supervisor._operator_lease_process_start_identity(pid)
    except Supervisor.SupervisorContractError:
        return None


def _terminate_exact_process(
    pid: int,
    process_start_identity_sha256: str,
    *,
    watchdog_started_session: bool = False,
) -> None:
    """Stop only the still-matching process identity; never a reused PID."""

    if _exact_process_identity(pid) != process_start_identity_sha256:
        return

    def send(selected_signal: int) -> None:
        if (
            watchdog_started_session
            and os.getpgid(pid) == pid
        ):
            os.killpg(pid, selected_signal)
        else:
            os.kill(pid, selected_signal)

    try:
        send(signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + WATCHDOG_TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _exact_process_identity(pid) != process_start_identity_sha256:
            return
        time.sleep(0.05)
    if _exact_process_identity(pid) == process_start_identity_sha256:
        try:
            send(signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_prior_owner(
    config: Orchestrator.OperatorConfiguration,
) -> None:
    """Adopt/reap a live operator left by a restarted watchdog service."""

    lease_root = (
        config.campaign_root
        / Supervisor.OPERATOR_LEASE_ROOT_NAME
    )
    current_path = lease_root / "current.json"
    if not (
        lease_root.exists()
        or lease_root.is_symlink()
        or current_path.exists()
        or current_path.is_symlink()
    ):
        return
    if (
        lease_root.is_symlink()
        or not lease_root.is_dir()
        or current_path.is_symlink()
        or not current_path.is_file()
    ):
        raise WatchdogError(
            "existing operator lease control is incomplete or unsafe"
        )
    try:
        observed = Supervisor.OperatorLease.observe_current(
            config.campaign_root,
            operator_configuration_sha256=config.config_sha256,
        )
    except Supervisor.SupervisorContractError as exc:
        raise WatchdogError(
            "existing operator lease cannot be authenticated"
        ) from exc
    heartbeat = observed["heartbeat"]
    pid = int(heartbeat["owner_pid"])
    identity = str(
        heartbeat["owner_process_start_identity_sha256"]
    )
    last_heartbeat_at_ns = int(heartbeat["heartbeat_at_ns"])
    while _exact_process_identity(pid) == identity:
        try:
            observed = Supervisor.OperatorLease.observe_current(
                config.campaign_root,
                operator_configuration_sha256=config.config_sha256,
            )
        except Supervisor.SupervisorContractError:
            if time.time_ns() - last_heartbeat_at_ns > int(
                WATCHDOG_HEARTBEAT_STALE_SECONDS
                * 1_000_000_000
            ):
                _terminate_exact_process(pid, identity)
                return
            time.sleep(min(0.05, WATCHDOG_POLL_SECONDS))
            continue
        heartbeat = observed["heartbeat"]
        if (
            int(heartbeat["owner_pid"]) != pid
            or heartbeat["owner_process_start_identity_sha256"]
            != identity
        ):
            raise WatchdogError(
                "operator lease owner changed while its prior PID "
                "remained live"
            )
        age_ns = time.time_ns() - int(heartbeat["heartbeat_at_ns"])
        last_heartbeat_at_ns = int(heartbeat["heartbeat_at_ns"])
        if age_ns < 0:
            raise WatchdogError(
                "operator heartbeat time is from the future"
            )
        if age_ns > int(
            WATCHDOG_HEARTBEAT_STALE_SECONDS * 1_000_000_000
        ):
            _terminate_exact_process(pid, identity)
            return
        time.sleep(WATCHDOG_POLL_SECONDS)


def _monitor_started_operator(
    config: Orchestrator.OperatorConfiguration,
    process: Any,
    *,
    started_identity: str,
    sleeper: Any,
    monotonic: Any = time.monotonic,
    wall_time_ns: Any = time.time_ns,
) -> int:
    """Observe startup, active heartbeat, and bounded released-process exit.

    The startup deadline applies only until this exact child publishes its
    first authenticated ACTIVE lease.  Once acquired, a transient read error
    is governed by heartbeat staleness—not by the long-expired startup
    deadline.  RELEASED is a bounded normal-shutdown phase so a multi-hour
    operator is not killed between releasing its lease and emitting terminal
    stdout.
    """

    startup_deadline = (
        monotonic() + WATCHDOG_STARTUP_TIMEOUT_SECONDS
    )
    observed_active = False
    last_heartbeat_at_ns: int | None = None
    released_deadline: float | None = None
    while process.poll() is None:
        try:
            observed = Supervisor.OperatorLease.observe_current(
                config.campaign_root,
                operator_configuration_sha256=config.config_sha256,
            )
        except Supervisor.SupervisorContractError:
            observed = None
        now_monotonic = monotonic()
        now_ns = wall_time_ns()
        if observed is not None:
            heartbeat = observed["heartbeat"]
            exact_owner = (
                heartbeat["owner_pid"] == process.pid
                and heartbeat[
                    "owner_process_start_identity_sha256"
                ]
                == started_identity
            )
            if exact_owner and heartbeat["status"] == "ACTIVE":
                heartbeat_at_ns = int(
                    heartbeat["heartbeat_at_ns"]
                )
                age_ns = now_ns - heartbeat_at_ns
                if age_ns < 0:
                    _terminate_exact_process(
                        process.pid,
                        started_identity,
                        watchdog_started_session=True,
                    )
                    raise WatchdogError(
                        "operator heartbeat is from the future"
                    )
                observed_active = True
                last_heartbeat_at_ns = heartbeat_at_ns
                released_deadline = None
                if age_ns > int(
                    WATCHDOG_HEARTBEAT_STALE_SECONDS
                    * 1_000_000_000
                ):
                    _terminate_exact_process(
                        process.pid,
                        started_identity,
                        watchdog_started_session=True,
                    )
            elif (
                exact_owner
                and heartbeat["status"] == "RELEASED"
                and observed_active
            ):
                if released_deadline is None:
                    released_deadline = (
                        now_monotonic
                        + WATCHDOG_RELEASE_EXIT_GRACE_SECONDS
                    )
                elif now_monotonic > released_deadline:
                    _terminate_exact_process(
                        process.pid,
                        started_identity,
                        watchdog_started_session=True,
                    )
            elif observed_active:
                _terminate_exact_process(
                    process.pid,
                    started_identity,
                    watchdog_started_session=True,
                )
                raise WatchdogError(
                    "operator lease owner changed while its child "
                    "remained live"
                )
            elif now_monotonic > startup_deadline:
                _terminate_exact_process(
                    process.pid,
                    started_identity,
                    watchdog_started_session=True,
                )
        elif observed_active:
            if (
                last_heartbeat_at_ns is None
                or now_ns - last_heartbeat_at_ns
                > int(
                    WATCHDOG_HEARTBEAT_STALE_SECONDS
                    * 1_000_000_000
                )
            ):
                _terminate_exact_process(
                    process.pid,
                    started_identity,
                    watchdog_started_session=True,
                )
        elif now_monotonic > startup_deadline:
            _terminate_exact_process(
                process.pid,
                started_identity,
                watchdog_started_session=True,
            )
        sleeper(WATCHDOG_POLL_SECONDS)
    return int(process.wait())


def _publish_restart_exhaustion(
    config: Orchestrator.OperatorConfiguration,
) -> dict[str, Any]:
    incident_path = config.campaign_root / "operator_incident.json"
    with Supervisor.OperatorLease(
        config.campaign_root,
        operator_configuration_sha256=config.config_sha256,
    ):
        if incident_path.exists() and not incident_path.is_symlink():
            return Orchestrator._strict_json(
                Orchestrator._read_regular(
                    incident_path,
                    maximum=Orchestrator.MAX_JSON_BYTES,
                ),
                label="operator incident receipt",
            )
        incident = Orchestrator._operator_incident_value(
            config,
            reason_code="watchdog_restart_exhausted",
            error_class="OperatorProcessFailure",
            runner_incident={
                "schema": 1,
                "kind":
                    "arc_agi3_contiguous_watchdog_intervention_request",
                "authority": "sealed_watchdog_restart_circuit",
                "human_intervention_required": True,
                "request":
                    "inspect authenticated watchdog/operator receipts "
                    "before a new service epoch",
            },
        )
        control_root = Orchestrator._canary_control_root(
            config.campaign_root
        )
        if (
            (control_root / "master_escrow.json").is_file()
            and (control_root / "placement_receipt.json").is_file()
        ):
            planting = Orchestrator._load_or_create_canary_planting(
                config
            )
            return Orchestrator._finalize_operator_terminal(
                campaign_root=config.campaign_root,
                planting=planting,
                terminal_receipt_path=incident_path,
                terminal_value=incident,
            )
        Orchestrator._ensure_receipt(incident_path, incident)
        return incident


def _open_private_stream(path: Path) -> Any:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise WatchdogError(
            "watchdog stream path appeared or is unsafe"
        ) from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise WatchdogError(
            "watchdog stream is not an unaliased owner-held file"
        )
    return os.fdopen(descriptor, "wb")


def _existing_invocations(
    root: Path,
) -> tuple[tuple[int, Path], ...]:
    rows: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if path.name in {"watchdog.lock", "runtime"}:
            continue
        matched = re.fullmatch(r"invocation_([0-9]{2})", path.name)
        if (
            matched is None
            or path.is_symlink()
            or not path.is_dir()
        ):
            raise WatchdogError(
                "watchdog root contains an unexpected entry"
            )
        rows.append((int(matched.group(1)), path))
    rows.sort()
    if [index for index, _path in rows] != list(range(len(rows))):
        raise WatchdogError(
            "watchdog invocation history is not contiguous"
        )
    return tuple(rows)


def _recover_terminal_invocation(
    config: Orchestrator.OperatorConfiguration,
    rows: Sequence[tuple[int, Path]],
) -> dict[str, Any] | None:
    if not rows:
        return None
    _index, latest = rows[-1]
    stdout_path = latest / "stdout.json"
    stderr_path = latest / "stderr.bin"
    if (
        stdout_path.is_symlink()
        or stderr_path.is_symlink()
        or not stdout_path.is_file()
        or not stderr_path.is_file()
    ):
        return None
    stdout_raw = Supervisor._read_regular_bytes(stdout_path)
    stderr_raw = Supervisor._read_regular_bytes(stderr_path)
    if stderr_raw:
        return None
    try:
        value = _strict_terminal(stdout_raw)
        return _validate_terminal_authority(
            config,
            value,
            returncode=(
                2
                if value["status"]
                in {
                    "OPERATOR_INCIDENT",
                    "PREFLIGHT_FAILED",
                    "JOURNAL_OR_STORAGE_EXHAUSTED",
                }
                else 0
            ),
        )
    except (WatchdogError, OSError):
        return None


def _seal_abandoned_invocations(
    config: Orchestrator.OperatorConfiguration,
    rows: Sequence[tuple[int, Path]],
    *,
    reason_code: str,
) -> None:
    """Make every crash-interrupted service invocation explicit and immutable."""

    if reason_code not in {
        "watchdog_restart_recovery",
        "operator_exit_without_terminal_authority",
    }:
        raise WatchdogError(
            "watchdog invocation recovery reason is invalid"
        )
    for index, invocation in rows:
        allowed = {"stdout.json", "stderr.bin", "recovery.json"}
        observed_names = {path.name for path in invocation.iterdir()}
        if not observed_names.issubset(allowed):
            raise WatchdogError(
                "watchdog invocation contains an unexpected entry"
            )
        stream_bindings: dict[str, dict[str, object]] = {}
        for name in ("stdout.json", "stderr.bin"):
            path = invocation / name
            if not (path.exists() or path.is_symlink()):
                with _open_private_stream(path):
                    pass
            if path.is_symlink() or not path.is_file():
                raise WatchdogError(
                    "abandoned watchdog stream is unsafe"
                )
            raw = Supervisor._read_regular_bytes(path)
            if len(raw) > WATCHDOG_MAX_STREAM_BYTES:
                raise WatchdogError(
                    "abandoned watchdog stream is oversized"
                )
            path.chmod(0o400)
            stream_bindings[name] = {
                "sha256": Orchestrator._sha256(raw),
                "bytes": len(raw),
            }
        body = {
            "schema": WATCHDOG_SCHEMA,
            "kind":
                "arc_agi3_contiguous_watchdog_nonterminal_invocation",
            "status": "NONTERMINAL_SEALED",
            "reason_code": reason_code,
            "operator_configuration_sha256":
                config.config_sha256,
            "invocation_index": index,
            "candidate_authority": False,
            "wip_authority": False,
            "promotion_authority": False,
            "streams": stream_bindings,
        }
        value = {
            **body,
            "receipt_sha256":
                Orchestrator._json_sha256(body),
        }
        recovery = invocation / "recovery.json"
        raw = _canonical_json(value) + b"\n"
        if recovery.exists() or recovery.is_symlink():
            if recovery.is_symlink():
                raise WatchdogError(
                    "abandoned invocation recovery receipt differs"
                )
            retained_raw = Supervisor._read_regular_bytes(recovery)
            try:
                retained = json.loads(
                    retained_raw.decode("ascii")
                )
            except (UnicodeError, ValueError) as exc:
                raise WatchdogError(
                    "abandoned invocation recovery receipt differs"
                ) from exc
            if not isinstance(retained, dict):
                raise WatchdogError(
                    "abandoned invocation recovery receipt differs"
                )
            retained_body = {
                key: item
                for key, item in retained.items()
                if key != "receipt_sha256"
            }
            if (
                _canonical_json(retained) + b"\n"
                != retained_raw
                or retained_body.get("schema") != WATCHDOG_SCHEMA
                or retained_body.get("kind")
                != (
                    "arc_agi3_contiguous_watchdog_"
                    "nonterminal_invocation"
                )
                or retained_body.get("status")
                != "NONTERMINAL_SEALED"
                or retained_body.get("reason_code")
                not in {
                    "watchdog_restart_recovery",
                    "operator_exit_without_terminal_authority",
                }
                or retained_body.get(
                    "operator_configuration_sha256"
                )
                != config.config_sha256
                or retained_body.get("invocation_index") != index
                or retained_body.get("streams") != stream_bindings
                or any(
                    retained_body.get(name) is not False
                    for name in (
                        "candidate_authority",
                        "wip_authority",
                        "promotion_authority",
                    )
                )
                or retained.get("receipt_sha256")
                != Orchestrator._json_sha256(retained_body)
            ):
                raise WatchdogError(
                    "abandoned invocation recovery receipt differs"
                )
        else:
            with _open_private_stream(recovery) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            recovery.chmod(0o400)
        Supervisor._fsync_directory(invocation)


def run_watchdog(
    config: Orchestrator.OperatorConfiguration,
    *,
    process_factory: Any = subprocess.Popen,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    """Run the sealed restart circuit until a terminal operator receipt."""

    _production_preflight(config)
    Orchestrator._ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    with WatchdogLease(config.campaign_root):
        _wait_for_prior_owner(config)
        root = config.campaign_root / WATCHDOG_ROOT_NAME
        prior_invocations = _existing_invocations(root)
        recovered = _recover_terminal_invocation(
            config, prior_invocations
        )
        if recovered is not None:
            return recovered
        _seal_abandoned_invocations(
            config,
            prior_invocations,
            reason_code="watchdog_restart_recovery",
        )
        first_restart_index = len(prior_invocations)
        if first_restart_index >= WATCHDOG_MAX_RESTARTS:
            return _publish_restart_exhaustion(config)
        environment = _operator_environment(root)
        command = _operator_command(config)
        for restart_index in range(
            first_restart_index, WATCHDOG_MAX_RESTARTS
        ):
            invocation = root / f"invocation_{restart_index:02d}"
            Supervisor._operator_lease_private_directory(
                invocation,
                create=True,
                label="watchdog invocation root",
            )
            stdout_path = invocation / "stdout.json"
            stderr_path = invocation / "stderr.bin"
            with (
                _open_private_stream(stdout_path) as stdout,
                _open_private_stream(stderr_path) as stderr,
            ):
                process = process_factory(
                    command,
                    cwd=str(config.runtime_control_snapshot_root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    close_fds=True,
                    start_new_session=True,
                    preexec_fn=_limit_operator_child,
                )
                started_identity = _exact_process_identity(process.pid)
                if started_identity is None:
                    raise WatchdogError(
                        "operator child lacks a process-start identity"
                    )
                returncode = _monitor_started_operator(
                    config,
                    process,
                    started_identity=started_identity,
                    sleeper=sleeper,
                )
            os.chmod(stdout_path, 0o400)
            os.chmod(stderr_path, 0o400)
            stdout_raw = stdout_path.read_bytes()
            stderr_raw = stderr_path.read_bytes()
            if (
                len(stdout_raw) > WATCHDOG_MAX_STREAM_BYTES
                or len(stderr_raw) > WATCHDOG_MAX_STREAM_BYTES
            ):
                raise WatchdogError(
                    "operator stream exceeded watchdog bound"
                )
            if not stderr_raw:
                try:
                    terminal = _validate_terminal_authority(
                        config,
                        _strict_terminal(stdout_raw),
                        returncode=returncode,
                    )
                except WatchdogError:
                    terminal = None
                if terminal is not None:
                    return terminal
            _seal_abandoned_invocations(
                config,
                ((restart_index, invocation),),
                reason_code=(
                    "operator_exit_without_terminal_authority"
                ),
            )
            if restart_index + 1 < WATCHDOG_MAX_RESTARTS:
                sleeper(
                    WATCHDOG_RESTART_BACKOFF_SECONDS[
                        restart_index + 1
                    ]
                )
            del returncode
        return _publish_restart_exhaustion(config)


class _SingleUseConfigAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del option_string
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--config may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed ARC-AGI-3 contiguous operator service",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        default=None,
        action=_SingleUseConfigAction,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_watchdog(
            Orchestrator.load_operator_configuration(args.config)
        )
    except Exception as exc:
        error_class = type(exc).__name__
        if re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_class
        ) is None:
            error_class = "WatchdogFailure"
        body = {
            "schema": WATCHDOG_SCHEMA,
            "kind": "arc_agi3_contiguous_watchdog_preflight",
            "status": "PREFLIGHT_FAILED",
            "reason_code": "watchdog_preflight_failed",
            "error_class": error_class,
        }
        result = {
            **body,
            "receipt_sha256":
                Orchestrator._json_sha256(body),
        }
        sys.stdout.buffer.write(_canonical_json(result) + b"\n")
        return 2
    sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    return (
        2
        if result.get("status")
        in {
            "OPERATOR_INCIDENT",
            "PREFLIGHT_FAILED",
            "JOURNAL_OR_STORAGE_EXHAUSTED",
        }
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
