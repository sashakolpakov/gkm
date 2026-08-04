from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import hmac
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

import arc_agi3_container_backend as B
import arc_agi3_codex_app_server_transport as T
import arc_agi3_containment_canary_operator as CanaryOperator
import arc_agi3_contiguous_runner as Runner
import arc_agi3_contiguous_taint as Taint


MANIFEST_A = "sha256:" + "a" * 64
MANIFEST_B = "sha256:" + "b" * 64
IMAGE_ID_A = "sha256:" + "1" * 64
IMAGE_ID_B = "sha256:" + "2" * 64
CONTAINER_ID = "3" * 64
IMAGE_REFERENCE = f"registry.example/gkm/arc-worker@{MANIFEST_A}"
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_TEMP_PREFIXES = (
    ".a3vr_",
    ".a3cb_rpc_",
    ".a3cb_ctl_",
    ".a3cb_stream_",
    ".a3cb_probe_comma_,",
    ".a3d_",
)


def _cleanup_private_test_root(path: Path) -> None:
    selected = Path(path)
    if not selected.exists() and not selected.is_symlink():
        return
    if not selected.is_symlink():
        os.chmod(selected, 0o700, follow_symlinks=False)
        for current, directories, files in os.walk(
            selected, topdown=True, followlinks=False
        ):
            os.chmod(current, 0o700, follow_symlinks=False)
            for directory in directories:
                child = Path(current) / directory
                if not child.is_symlink():
                    os.chmod(child, 0o700, follow_symlinks=False)
            for filename in files:
                child = Path(current) / filename
                if not child.is_symlink():
                    os.chmod(child, 0o600, follow_symlinks=False)

    def repair_and_retry(function, name, _error):
        observed = Path(name)
        if observed.is_symlink():
            observed.unlink()
            return
        os.chmod(observed, 0o700, follow_symlinks=False)
        function(name)

    shutil.rmtree(selected, onerror=repair_and_retry)


@contextlib.contextmanager
def _system_private_test_root(prefix: str):
    # Keep AF_UNIX paths short without ever allocating inside the checkout.
    root = Path(
        tempfile.mkdtemp(
            prefix=prefix,
            dir=Path(tempfile.gettempdir()).resolve(),
        )
    )
    root.chmod(0o700)
    try:
        yield root
    finally:
        _cleanup_private_test_root(root)


def _request_private_test_root(
    request: pytest.FixtureRequest,
    *,
    prefix: str,
) -> Path:
    manager = _system_private_test_root(prefix)
    root = manager.__enter__()
    request.addfinalizer(
        lambda: manager.__exit__(None, None, None)
    )
    return root


def _request_short_private_test_root(
    request: pytest.FixtureRequest,
) -> Path:
    configured = os.environ.get("ARC_AGI3_TEST_SHORT_TMPDIR")
    checkout = _CHECKOUT_ROOT
    if configured is not None:
        candidates = (Path(configured),)
    else:
        # macOS commonly exposes a long per-user TMPDIR even though its
        # private system temp root is available.  Select the shortest safe
        # resolved temp base so AF_UNIX tests pass without hidden shell setup.
        candidates = tuple({
            Path(tempfile.gettempdir()).resolve(),
            Path("/private/tmp"),
            Path("/tmp").resolve(),
        })
    valid_bases: list[Path] = []
    for candidate in candidates:
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_dir()
            or not os.access(candidate, os.W_OK | os.X_OK)
        ):
            continue
        resolved = candidate.resolve()
        if resolved == checkout or checkout in resolved.parents:
            continue
        valid_bases.append(resolved)
    if not valid_bases:
        raise RuntimeError(
            "short private test root base is unavailable"
        )
    resolved_base = min(
        set(valid_bases),
        key=lambda path: (len(os.fsencode(path)), str(path)),
    )
    root: Path | None = None
    for _ in range(32):
        candidate = resolved_base / ("a" + uuid.uuid4().hex[:2])
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        root = candidate
        break
    if root is None:
        raise RuntimeError("could not allocate a short private test root")
    request.addfinalizer(lambda: _cleanup_private_test_root(root))
    return root


def _repo_root_temp_inventory() -> tuple[str, ...]:
    return tuple(sorted(
        child.name
        for child in _CHECKOUT_ROOT.iterdir()
        if (
            child.name.startswith(_REPO_ROOT_TEMP_PREFIXES)
            or re.fullmatch(r"a[0-9a-f]{2}", child.name)
            is not None
        )
    ))


def test_system_private_temp_cleanup_never_leaks_into_repository_root():
    before = _repo_root_temp_inventory()

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with _system_private_test_root(".a3cb_rpc_") as root:
            assert _CHECKOUT_ROOT not in root.parents
            sealed = root / "sealed"
            sealed.mkdir(mode=0o700)
            evidence = sealed / "evidence.json"
            evidence.write_text("{}\n", encoding="ascii")
            evidence.chmod(0o400)
            sealed.chmod(0o500)
            root.chmod(0o500)
            raise RuntimeError("deliberate failure")

    with _system_private_test_root(".a3cb_stream_") as root:
        read_only = root / "read-only"
        read_only.mkdir(mode=0o700)
        (read_only / "receipt").write_bytes(b"immutable\n")
        (read_only / "receipt").chmod(0o400)
        read_only.chmod(0o500)
        root.chmod(0o500)

    assert _repo_root_temp_inventory() == before


def test_checkout_has_no_legacy_raw_quarantine_retention_root():
    forbidden = (
        _CHECKOUT_ROOT
        / "arc"
        / "crack_lab"
        / "quarantined_attempts"
    )
    assert not forbidden.exists()
    assert not forbidden.is_symlink()


def test_r8_path_alias_sqlite_warning_is_typed_deterministic_configuration():
    message = (
        "WARNING: proceeding, even though we could not create PATH aliases: "
        "Operation not permitted (os error 1); Error: failed to initialize "
        "sqlite state runtime under /Users/sasha/.codex"
    )
    error = T.DeterministicControllerConfigurationError(
        message,
        failure_code="controller_path_alias_or_startup_stderr",
    )

    assert B._classify_substrate_preflight_failure(
        "controller-start-and-initialize",
        error,
    ) == (
        "DETERMINISTIC_CONFIGURATION",
        "controller_path_alias_or_startup_stderr",
    )


def _native_workspace_receipt() -> dict[str, Any]:
    return {
        "policy": "isolated-local-git-root-no-parent-discovery-v1",
        "workspace_root": B.CONTROLLER_NEUTRAL_DESTINATION,
        "git_dir": f"{B.CONTROLLER_NEUTRAL_DESTINATION}/.git",
        "git_root_equals_workspace": True,
        "head_ref": "refs/heads/contiguous",
        "head_commit": "f" * 40,
        "file_count": 6,
        "inventory_sha256": "e" * 64,
        "symlink_count": 0,
        "hardlink_count": 0,
        "path_escape_count": 0,
        "forbidden_classes_absent": [
            "campaign-plan",
            "sidecar-or-quarantine-output",
            "manuscript",
            "comparator",
            "benchmark",
            "parent-repository-git-metadata",
        ],
        "git_ceiling_directories": B.CONTROLLER_NEUTRAL_DESTINATION,
        "git_discovery_across_filesystem": False,
        "git_global_config_disabled": True,
        "git_system_config_disabled": True,
    }


def test_controller_native_workspace_receipt_is_strict_and_fail_closed():
    valid = _native_workspace_receipt()
    B.DockerControllerContainerLauncher._validate_guardian_native_workspace(
        valid
    )
    for field, replacement in (
        ("workspace_root", "/parent/repository"),
        ("git_dir", "/parent/repository/.git"),
        ("git_root_equals_workspace", False),
        ("symlink_count", 1),
        ("hardlink_count", 1),
        ("path_escape_count", 1),
        ("forbidden_classes_absent", []),
        ("git_ceiling_directories", "/"),
        ("git_discovery_across_filesystem", True),
    ):
        tampered = {**valid, field: replacement}
        with pytest.raises(
            B.ContainerContractError,
            match="workspace proof",
        ):
            (
                B.DockerControllerContainerLauncher
                ._validate_guardian_native_workspace(tampered)
            )


def test_controller_egress_probe_precedes_controller_creation_and_fails_closed(
    tmp_path,
):
    events: list[str] = []
    proxy_id = "1" * 64
    controller_id = "2" * 64

    class BackendDouble:
        def _required(self, argv, **_kwargs):
            events.append(argv[-1])
            if argv[-1] == "proxy-create":
                return B.CommandResult(tuple(argv), 0, proxy_id + "\n")
            if argv[-1] == "controller-create":
                return B.CommandResult(
                    tuple(argv), 0, controller_id + "\n"
                )
            return B.CommandResult(tuple(argv), 0)

        def _inspect_container(self, container):
            events.append(f"inspect:{container[0]}")
            if container == controller_id:
                return {
                    "container": container,
                    "Config": {
                        "Cmd": list(B.CONTROLLER_CHILD_COMMAND)
                    },
                }
            return {"container": container}

    launcher = B.DockerControllerContainerLauncher.__new__(
        B.DockerControllerContainerLauncher
    )
    launcher._backend = BackendDouble()
    launcher._docker = "docker"
    launcher._proxy_create_argv = (
        lambda **_kwargs: ("docker", "proxy-create")
    )
    launcher._controller_create_argv = (
        lambda **_kwargs: ("docker", "controller-create")
    )
    launcher._validate_role_container = (
        lambda **kwargs: (
            events.append(f"validate:{kwargs['role']}") or "a" * 64
        )
    )
    launcher._ensure_proxy_ready = (
        lambda **_kwargs: (
            events.append("egress-live-probe")
            or (
                tmp_path / "ready.json",
                "b" * 64,
                tmp_path / "probe.json",
                "c" * 64,
            )
        )
    )
    launcher._remove_role = (
        lambda *_args, **kwargs: events.append(
            f"remove:{kwargs['role']}"
        )
    )
    binding = SimpleNamespace(
        campaign_id=str(uuid.uuid4()),
        generation_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        attempt_spec_sha256="d" * 64,
    )
    transport = SimpleNamespace(
        controller_image_digest=MANIFEST_A,
        controller_egress_proxy_image_digest=MANIFEST_B,
        controller_egress_policy="openai_https_only",
        controller_egress_policy_sha256="e" * 64,
        controller_user="65532:65532",
        controller_entrypoint=(
            "/usr/local/bin/arc-agi3-contiguous-controller-guardian",
        ),
    )
    image = SimpleNamespace()

    result = launcher._create_roles(
        binding=binding,
        transport=transport,
        controller_image=image,
        proxy_image=image,
        state_root=tmp_path,
    )
    assert result[:2] == (controller_id, proxy_id)
    assert events.index("egress-live-probe") < events.index(
        "controller-create"
    )

    events.clear()

    def fail_ready(**_kwargs):
        events.append("egress-live-probe")
        raise B.ContainerContractError("deny probe failed")

    launcher._ensure_proxy_ready = fail_ready
    with pytest.raises(B.ContainerContractError, match="deny probe"):
        launcher._create_roles(
            binding=binding,
            transport=transport,
            controller_image=image,
            proxy_image=image,
            state_root=tmp_path,
        )
    assert "controller-create" not in events
    assert events[-1] == f"remove:{B.EGRESS_PROXY_ROLE}"


def test_container_tmpfs_receipt_uses_bounded_exec_and_is_immutable(
    tmp_path,
):
    root = tmp_path / "host"
    root.mkdir(mode=0o700)
    receipt = root / "ready.json"
    container_id = "3" * 64
    payload = b'{"kind":"ready","status":"PASS"}\n'
    commands: list[tuple[str, ...]] = []

    class RunnerDouble:
        def run(self, argv, **_kwargs):
            command = tuple(argv)
            commands.append(command)
            return B.CommandResult(command, 0, payload.decode("ascii"))

    launcher = B.DockerControllerContainerLauncher.__new__(
        B.DockerControllerContainerLauncher
    )
    launcher._docker = "docker"
    launcher._runner = RunnerDouble()

    digest = launcher._copy_ephemeral_receipt(
        container_id=container_id,
        container_path="/run/receipt.json",
        host_path=receipt,
        timeout_seconds=1,
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert receipt.read_bytes() == payload
    assert receipt.stat().st_mode & 0o777 == 0o400
    assert commands == [
        (
            "docker",
            "container",
            "exec",
            "--user",
            "0:0",
            container_id,
            "/bin/cat",
            "/run/receipt.json",
        )
    ]
    with pytest.raises(
        B.ContainerContractError, match="differs from the terminal stream"
    ):
        launcher._runner = SimpleNamespace(
            run=lambda argv, **_kwargs: B.CommandResult(
                tuple(argv), 0, '{"kind":"substituted"}\n'
            )
        )
        launcher._copy_ephemeral_receipt(
            container_id=container_id,
            container_path="/run/receipt.json",
            host_path=receipt,
            timeout_seconds=1,
        )


def test_backend_uses_runner_terminal_precedence_byte_exactly():
    for terminal_status in ("exited", "containment_fault"):
        for result_kind in (
            "clean_no_progress",
            "tainted",
            "infrastructure",
            "candidate",
            "blocker",
        ):
            result = Runner.AttemptResult(
                kind=result_kind,
                cost_used=7.25,
                reason=f"synthetic {result_kind}",
            )
            expected = Runner.apply_terminal_result_precedence(
                terminal_status,
                result,
            )
            observed = B._apply_terminal_result_precedence(
                Runner,
                terminal_status=terminal_status,
                result=result,
            )
            assert dataclasses.asdict(observed) == (
                dataclasses.asdict(expected)
            )


def test_stopped_controller_launch_is_archived_and_exact_roles_removed(
    tmp_path,
):
    host = tmp_path / "host"
    control = host / "app_server_control"
    control.mkdir(parents=True, mode=0o700)
    os.chmod(host, 0o700)
    os.chmod(control, 0o700)
    launch_path = host / "controller_launch_receipt.json"
    guardian_path = host / "controller_guardian_start.json"
    launch_path.write_text('{"launch":"old"}\n', encoding="utf-8")
    guardian_path.write_text('{"guardian":"old"}\n', encoding="utf-8")
    os.chmod(launch_path, 0o600)
    os.chmod(guardian_path, 0o600)

    controller_id = "1" * 64
    proxy_id = "2" * 64
    present = {
        B.CONTROLLER_ROLE: [controller_id],
        B.EGRESS_PROXY_ROLE: [proxy_id],
    }
    removed: list[tuple[str, str, bool]] = []
    launcher = B.DockerControllerContainerLauncher.__new__(
        B.DockerControllerContainerLauncher
    )
    launcher._query_role_ids = lambda _binding, role: tuple(
        present[role]
    )

    def remove(_binding, *, role, container_id, force):
        removed.append((role, container_id, force))
        present[role].clear()

    launcher._remove_role = remove
    binding = SimpleNamespace(app_server_control_dir=str(control))
    launcher._supersede_stopped_launch(
        binding=binding,
        controller_id=controller_id,
        proxy_id=proxy_id,
        launch_path=launch_path,
    )

    assert removed == [
        (B.CONTROLLER_ROLE, controller_id, True),
        (B.EGRESS_PROXY_ROLE, proxy_id, True),
    ]
    assert not launch_path.exists()
    assert not guardian_path.exists()
    assert len(list(host.glob("controller_launch_receipt.superseded-*"))) == 1
    assert len(list(host.glob("controller_guardian_start.superseded-*"))) == 1


def test_stopped_controller_launch_recovery_rejects_wrong_role_id(
    tmp_path,
):
    host = tmp_path / "host"
    control = host / "app_server_control"
    control.mkdir(parents=True, mode=0o700)
    os.chmod(host, 0o700)
    os.chmod(control, 0o700)
    launch_path = host / "controller_launch_receipt.json"
    launch_path.write_text('{"launch":"old"}\n', encoding="utf-8")
    os.chmod(launch_path, 0o600)

    launcher = B.DockerControllerContainerLauncher.__new__(
        B.DockerControllerContainerLauncher
    )
    launcher._query_role_ids = (
        lambda _binding, role:
        ("f" * 64,)
        if role == B.CONTROLLER_ROLE
        else ("2" * 64,)
    )
    launcher._remove_role = lambda *args, **kwargs: pytest.fail(
        "ambiguous role must not be removed"
    )
    binding = SimpleNamespace(app_server_control_dir=str(control))
    with pytest.raises(
        B.ContainerContractError,
        match="recovery is ambiguous",
    ):
        launcher._supersede_stopped_launch(
            binding=binding,
            controller_id="1" * 64,
            proxy_id="2" * 64,
            launch_path=launch_path,
        )
    assert launch_path.exists()


def _image_record(
    *,
    manifest: str = MANIFEST_A,
    image_id: str = IMAGE_ID_A,
) -> dict[str, Any]:
    return {
        "Id": image_id,
        "RepoDigests": [f"registry.example/gkm/arc-worker@{manifest}"],
        "Config": {
            "Env": [
                "PATH=/usr/local/bin:/usr/bin:/bin",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PYTHONDONTWRITEBYTECODE=1",
                "PYTHONUNBUFFERED=1",
                "PYTHON_VERSION=3.12.11",
                "PYTHON_PIP_VERSION=25.1",
                "PYTHON_SETUPTOOLS_VERSION=80.9.0",
                "PYTHON_GET_PIP_URL=https://example.invalid/get-pip.py",
                "PYTHON_GET_PIP_SHA256=" + "4" * 64,
                "PYTHON_SHA256=" + "5" * 64,
                "GPG_KEY=public-build-key",
            ],
            "Entrypoint": [B.PYTHON_ENTRYPOINT, "-I"],
            "Labels": B.trusted_worker_hashes(),
        },
    }


def _container_record(
    spec: B.AttemptSpec,
    *,
    image_id: str = IMAGE_ID_A,
    container_id: str = CONTAINER_ID,
    running: bool = False,
    status: str | None = None,
) -> dict[str, Any]:
    identity = spec.identity
    limits = spec.resource_limits
    environment = _image_record()["Config"]["Env"] + [
        f"ARC_AGI3_CAMPAIGN_ID={identity.campaign_id}",
        f"ARC_AGI3_GENERATION_ID={identity.generation_id}",
        f"ARC_AGI3_ATTEMPT_ID={identity.attempt_id}",
        f"ARC_AGI3_GAME={identity.game}",
        f"ARC_AGI3_TARGET_LEVEL={identity.target_level}",
    ]
    mounts = [
        {
            "Type": "bind",
            "Source": str(spec.parent_input),
            "Destination": B.INPUT_DESTINATION,
            "RW": False,
            "Propagation": "rprivate",
        },
    ]
    if spec.role == "proposer":
        mounts.append(
            {
                "Type": "bind",
                "Source": str(spec.workspace_root),
                "Destination": B.WORKSPACE_DESTINATION,
                "RW": True,
                "Propagation": "rprivate",
            }
        )
    mounts.append(
        {
            "Type": "bind",
            "Source": str(spec.export_root),
            "Destination": B.EXPORT_DESTINATION,
            "RW": True,
            "Propagation": "rprivate",
        }
    )
    if spec.role == "proposer":
        mounts.append(
            {
                "Type": "bind",
                "Source": str(spec.bridge_root),
                "Destination": B.BRIDGE_ROOT_DESTINATION,
                "RW": True,
                "Propagation": "rprivate",
            }
        )
    if spec.role == "proposer":
        volume_name = B.arena_volume_name(spec.identity)
        mounts.append(
            {
                "Type": "volume",
                "Name": volume_name,
                "Driver": "local",
                "Source": (
                    f"/var/lib/docker/volumes/{volume_name}/_data"
                ),
                "Destination": B.PROPOSER_RPC_ROOT_DESTINATION,
                "RW": False,
                "Propagation": "",
            }
        )
    else:
        mounts.append(
            {
                "Type": "bind",
                "Source": str(spec.arena_socket),
                "Destination": B.RPC_SOCKET_DESTINATION,
                "RW": False,
                "Propagation": "rprivate",
            }
        )
    mounts.append(
        {
            "Type": "bind",
            "Source": str(spec.arena_token_file),
            "Destination": B.RPC_TOKEN_DESTINATION,
            "RW": False,
            "Propagation": "rprivate",
        }
    )
    if spec.role == "proposer":
        mounts.append(
            {
                "Type": "bind",
                "Source": str(spec.bridge_token_file),
                "Destination": B.BRIDGE_TOKEN_DESTINATION,
                "RW": False,
                "Propagation": "rprivate",
            }
        )
    return {
        "Id": container_id,
        "Image": image_id,
        "Name": "/" + (
            f"arc-agi3-{identity.game}-l{identity.target_level}-"
            f"{identity.generation_id[:8]}-{identity.attempt_id[:12]}"
        ),
        "Config": {
            "Image": spec.image_reference,
            "User": (
                f"{spec.export_root.stat().st_uid}:"
                f"{spec.export_root.stat().st_gid}"
            ),
            "Labels": {
                B.LABEL_CAMPAIGN: identity.campaign_id,
                B.LABEL_GENERATION: identity.generation_id,
                B.LABEL_ATTEMPT: identity.attempt_id,
                B.LABEL_GAME: identity.game,
                B.LABEL_LEVEL: str(identity.target_level),
                B.LABEL_ROLE: B.ATTEMPT_WORKER_ROLE,
                **B.trusted_worker_hashes(),
            },
            "Env": environment,
            "Cmd": list(spec.command),
            "Entrypoint": [B.PYTHON_ENTRYPOINT],
            "WorkingDir": (
                B.WORKSPACE_DESTINATION
                if spec.role == "proposer"
                else B.INPUT_DESTINATION
            ),
            "Healthcheck": {"Test": ["NONE"]},
        },
        "HostConfig": {
            "ReadonlyRootfs": True,
            "NetworkMode": "none",
            "PidMode": "",
            "CgroupnsMode": "private",
            "IpcMode": "private",
            "UTSMode": "",
            "Privileged": False,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "LogConfig": {
                "Type": "local",
                "Config": {"max-size": "4m", "max-file": "1"},
            },
            "CapAdd": [],
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges=true"],
            "NanoCpus": limits.nano_cpus,
            "Memory": limits.memory_bytes,
            "MemorySwap": limits.memory_bytes,
            "PidsLimit": limits.pids,
            "Devices": [],
            "DeviceRequests": [],
            "Tmpfs": {
                B.TMPFS_DESTINATION: (
                    "rw,nosuid,nodev,noexec,"
                    f"size={limits.tmpfs_bytes},mode=1777,"
                    f"uid={spec.export_root.stat().st_uid},"
                    f"gid={spec.export_root.stat().st_gid}"
                )
            },
        },
        "Mounts": mounts,
        "NetworkSettings": {"Networks": {}},
        "State": {
            "Status": status or ("running" if running else "created"),
            "Running": running,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "ExitCode": 0,
            "Error": "",
        },
    }


class FakeDockerRunner:
    """Stateful Docker CLI double; no test invokes Docker."""

    def __init__(
        self,
        spec: B.AttemptSpec,
        *,
        image_records: Sequence[dict[str, Any]] | None = None,
        container_mutator: Callable[[dict[str, Any]], None] | None = None,
        label_query_output: str | None = None,
        log_stdout: str = "",
        log_stderr: str = "",
    ) -> None:
        self.spec = spec
        self.image_records = list(image_records or [_image_record()])
        self.image_calls = 0
        self.container_mutator = container_mutator
        self.label_query_output = label_query_output
        self.log_stdout = log_stdout
        self.log_stderr = log_stderr
        self.commands: list[tuple[str, ...]] = []
        self.created = False
        self.started = False
        self.running = False
        self.removed = False
        self.arena_volume_created = False
        self.bridge_listener: socket.socket | None = None

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> B.CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        if command[:3] == ("docker", "image", "inspect"):
            index = min(self.image_calls, len(self.image_records) - 1)
            record = self.image_records[index]
            self.image_calls += 1
            return self._result(command, stdout=json.dumps([record]))
        if command[:3] == ("docker", "container", "create"):
            self.created = True
            self.started = False
            self.removed = False
            return self._result(command, stdout=CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "inspect"):
            if not self.created or self.removed:
                return self._result(command, returncode=1, stderr="absent")
            latest_image = self.image_records[
                min(max(self.image_calls - 1, 0), len(self.image_records) - 1)
            ]
            record = _container_record(
                self.spec,
                image_id=latest_image["Id"],
                running=self.running,
                status=(
                    "running"
                    if self.running
                    else ("exited" if self.started else "created")
                ),
            )
            if self.container_mutator is not None:
                self.container_mutator(record)
            return self._result(command, stdout=json.dumps([record]))
        if command[:3] == ("docker", "container", "start"):
            if not self.created or self.removed:
                return self._result(command, returncode=1)
            self.started = True
            self.running = True
            if (
                self.spec.role == "proposer"
                and self.spec.bridge_socket is not None
                and not self.spec.bridge_socket.exists()
            ):
                self.bridge_listener = socket.socket(
                    socket.AF_UNIX, socket.SOCK_STREAM
                )
                self.bridge_listener.bind(str(self.spec.bridge_socket))
                self.spec.bridge_socket.chmod(0o600)
                self.bridge_listener.listen(1)
            return self._result(command, stdout=CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "top"):
            if not self.created or self.removed or not self.running:
                return self._result(command, returncode=1, stderr="not running")
            return self._result(
                command,
                stdout="PID PPID\n4242 1\n4243 4242\n",
            )
        if command[:3] == ("docker", "container", "logs"):
            if not self.created or self.removed or self.running:
                return self._result(command, returncode=1, stderr="unavailable")
            return self._result(
                command,
                stdout=self.log_stdout,
                stderr=self.log_stderr,
            )
        if command[:3] == ("docker", "container", "stop"):
            self.running = False
            if self.bridge_listener is not None:
                self.bridge_listener.close()
                self.bridge_listener = None
            return self._result(command, stdout=CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "kill"):
            self.running = False
            if self.bridge_listener is not None:
                self.bridge_listener.close()
                self.bridge_listener = None
            return self._result(command, stdout=CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "rm"):
            self.running = False
            self.removed = True
            if self.bridge_listener is not None:
                self.bridge_listener.close()
                self.bridge_listener = None
            return self._result(command, stdout=CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "ls"):
            output = self.label_query_output
            if output is None:
                filters = {
                    command[index + 1]
                    for index, value in enumerate(command[:-1])
                    if value == "--filter"
                }
                expected_role = (
                    f"{B.LABEL_ROLE}={B.ATTEMPT_WORKER_ROLE}"
                )
                has_other_role = any(
                    value.startswith(f"{B.LABEL_ROLE}=")
                    and value != expected_role
                    for value in filters
                )
                output = (
                    CONTAINER_ID + "\n"
                    if (
                        self.created
                        and not self.removed
                        and not has_other_role
                    )
                    else ""
                )
            return self._result(command, stdout=output)
        if command[:3] == ("docker", "volume", "ls"):
            return self._result(
                command,
                stdout=(
                    B.arena_volume_name(self.spec.identity) + "\n"
                    if self.arena_volume_created
                    else ""
                ),
            )
        if command[:3] == ("docker", "volume", "inspect"):
            if not self.arena_volume_created:
                return self._result(
                    command, returncode=1, stderr="absent"
                )
            raise AssertionError(
                "generic Docker double has no live Arena volume record"
            )
        if command[:3] == ("docker", "volume", "rm"):
            self.arena_volume_created = False
            return self._result(
                command,
                stdout=B.arena_volume_name(self.spec.identity) + "\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    @staticmethod
    def _result(
        command: tuple[str, ...],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> B.CommandResult:
        return B.CommandResult(
            argv=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )


RELAY_CONTAINER_ID = "6" * 64
RELAY_IMAGE_ID = "sha256:" + "7" * 64
RELAY_MANIFEST = "sha256:" + "d" * 64
RELAY_REFERENCE = f"gkm/arc-arena-relay@{RELAY_MANIFEST}"


class _RelayAttachmentDouble:
    def __init__(self):
        self.relay_argv = (
            "docker",
            "container",
            "attach",
            RELAY_CONTAINER_ID,
        )
        self.arena_socket_identity_sha256 = "8" * 64
        self.aborted = False
        self.finished = False

    def finish(self, *, timeout_seconds):
        assert timeout_seconds == 30
        self.finished = True
        return {
            "schema": 1,
            "kind": "arc_agi3_attached_arena_relay",
            "status": "PASS",
            "relay_container_id": RELAY_CONTAINER_ID,
            "threads_stopped": True,
        }

    def abort(self):
        self.aborted = True


class _ArenaVolumeDockerDouble:
    def __init__(
        self,
        identity: B.AttemptIdentity,
        *,
        relay_mutator=None,
    ):
        self.identity = identity
        self.relay_mutator = relay_mutator
        self.commands: list[tuple[str, ...]] = []
        self.volume_created = False
        self.relay_created = False
        self.relay_running = False
        self.relay_removed = False
        self.readiness_nonce = ""

    @property
    def volume_name(self):
        return B.arena_volume_name(self.identity)

    @property
    def labels(self):
        return B.DockerArenaVolumeLifecycle._identity_labels(
            self.identity
        )

    @property
    def source_sha256(self):
        return hashlib.sha256(
            Path(B.__file__).with_name(
                "arc_agi3_arena_volume_relay.py"
            ).read_bytes()
        ).hexdigest()

    def _image(self):
        return {
            "Id": RELAY_IMAGE_ID,
            "RepoDigests": [RELAY_REFERENCE],
            "Config": {
                "Env": _image_record()["Config"]["Env"],
                "Entrypoint": [B.ARENA_RELAY_ENTRYPOINT],
                "User": "0:0",
                "WorkingDir": "/",
                "Labels": {
                    B.LABEL_ROLE: B.ARENA_VOLUME_RELAY_ROLE,
                    B.ARENA_RELAY_LABEL_TRANSPORT:
                        B.ARENA_VOLUME_TRANSPORT,
                    B.ARENA_RELAY_LABEL_SOURCE_SHA256:
                        self.source_sha256,
                },
            },
        }

    def _volume(self):
        return {
            "Name": self.volume_name,
            "Driver": "local",
            "Scope": "local",
            "Labels": self.labels,
            "Options": None,
            "Mountpoint": (
                f"/var/lib/docker/volumes/{self.volume_name}/_data"
            ),
        }

    def _relay(self):
        labels = {
            **self.labels,
            B.ARENA_RELAY_LABEL_TRANSPORT:
                B.ARENA_VOLUME_TRANSPORT,
            B.ARENA_RELAY_LABEL_SOURCE_SHA256:
                self.source_sha256,
        }
        value = {
            "Id": RELAY_CONTAINER_ID,
            "Image": RELAY_IMAGE_ID,
            "Name": "/" + B.arena_relay_container_name(self.identity),
            "Config": {
                "Image": RELAY_REFERENCE,
                "User": "0:0",
                "Entrypoint": [B.ARENA_RELAY_ENTRYPOINT],
                "Cmd": list(
                    B.DockerArenaVolumeLifecycle(
                        B.DockerContainerBackend(self)
                    )._relay_command(
                        self.identity,
                        readiness_nonce=self.readiness_nonce,
                    )
                ),
                "WorkingDir": "/",
                "Healthcheck": {"Test": ["NONE"]},
                "Labels": labels,
                "Env": _image_record()["Config"]["Env"],
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "NetworkMode": "none",
                "PidMode": "",
                "CgroupnsMode": "private",
                "IpcMode": "private",
                "UTSMode": "",
                "Privileged": False,
                "CapAdd": [],
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges=true"],
                "NanoCpus": int(
                    B.ARENA_RELAY_CPUS * 1_000_000_000
                ),
                "Memory": B.ARENA_RELAY_MEMORY_BYTES,
                "MemorySwap": B.ARENA_RELAY_MEMORY_BYTES,
                "PidsLimit": B.ARENA_RELAY_PIDS,
                "Tmpfs": {
                    B.ARENA_RELAY_TMPFS_DESTINATION: (
                        "rw,nosuid,nodev,noexec,"
                        f"size={B.ARENA_RELAY_TMPFS_BYTES},"
                        "mode=0700,uid=0,gid=0"
                    )
                },
                "RestartPolicy": {
                    "Name": "no",
                    "MaximumRetryCount": 0,
                },
                "LogConfig": {
                    "Type": "local",
                    "Config": {
                        "max-size": "1m",
                        "max-file": "1",
                    },
                },
                "Devices": [],
                "DeviceRequests": [],
            },
            "Mounts": [
                {
                    "Type": "volume",
                    "Name": self.volume_name,
                    "Driver": "local",
                    "Source": (
                        "/var/lib/docker/volumes/"
                        f"{self.volume_name}/_data"
                    ),
                    "Destination":
                        B.PROPOSER_RPC_ROOT_DESTINATION,
                    "RW": True,
                    "Propagation": "",
                }
            ],
            "NetworkSettings": {"Networks": {}},
            "State": {"Running": self.relay_running},
        }
        if self.relay_mutator is not None:
            self.relay_mutator(value)
        return value

    def run(self, argv, *, timeout_seconds=None):
        del timeout_seconds
        command = tuple(argv)
        self.commands.append(command)
        if command[:3] == ("docker", "container", "ls"):
            output = (
                RELAY_CONTAINER_ID + "\n"
                if self.relay_created and not self.relay_removed
                else ""
            )
            return B.CommandResult(command, 0, output)
        if command[:3] == ("docker", "volume", "ls"):
            output = self.volume_name + "\n" if self.volume_created else ""
            return B.CommandResult(command, 0, output)
        if command[:3] == ("docker", "image", "inspect"):
            return B.CommandResult(
                command, 0, json.dumps([self._image()])
            )
        if command[:3] == ("docker", "volume", "create"):
            self.volume_created = True
            return B.CommandResult(command, 0, self.volume_name + "\n")
        if command[:3] == ("docker", "volume", "inspect"):
            if not self.volume_created:
                return B.CommandResult(command, 1, "", "absent")
            return B.CommandResult(
                command, 0, json.dumps([self._volume()])
            )
        if command[:3] == ("docker", "volume", "rm"):
            self.volume_created = False
            return B.CommandResult(command, 0, self.volume_name + "\n")
        if command[:3] == ("docker", "container", "create"):
            nonce_arg = next(
                item
                for item in command
                if item.startswith("--readiness-nonce=")
            )
            self.readiness_nonce = nonce_arg.split("=", 1)[1]
            self.relay_created = True
            self.relay_removed = False
            return B.CommandResult(
                command, 0, RELAY_CONTAINER_ID + "\n"
            )
        if command[:3] == ("docker", "container", "inspect"):
            if not self.relay_created or self.relay_removed:
                return B.CommandResult(command, 1, "", "absent")
            return B.CommandResult(
                command, 0, json.dumps([self._relay()])
            )
        if command[:3] == ("docker", "container", "start"):
            self.relay_running = True
            return B.CommandResult(
                command, 0, RELAY_CONTAINER_ID + "\n"
            )
        if command[:3] == ("docker", "container", "exec"):
            readiness = {
                "schema": 1,
                "kind": "arc_agi3_arena_volume_relay_readiness",
                "status": "READY",
                "campaign_id": self.identity.campaign_id,
                "generation_id": self.identity.generation_id,
                "attempt_id": self.identity.attempt_id,
                "readiness_nonce": self.readiness_nonce,
                "relay_pid": 1,
                "socket_path": B.PROPOSER_RPC_SOCKET_DESTINATION,
                "socket_mode": 0o666,
                "network_mode_required": "none",
                "transport": B.ARENA_VOLUME_TRANSPORT,
            }
            return B.CommandResult(
                command,
                0,
                json.dumps(
                    readiness,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )
        if command[:3] == ("docker", "container", "rm"):
            self.relay_running = False
            self.relay_removed = True
            return B.CommandResult(
                command, 0, RELAY_CONTAINER_ID + "\n"
            )
        if command[:3] == ("docker", "container", "top"):
            if self.relay_removed:
                return B.CommandResult(command, 1, "", "absent")
            return B.CommandResult(command, 0, "PID PPID\n1 0\n")
        raise AssertionError(f"unexpected relay command: {command}")


def _arena_volume_transport():
    return SimpleNamespace(
        arena_transport=B.ARENA_VOLUME_TRANSPORT,
        arena_relay_image_reference=RELAY_REFERENCE,
        arena_relay_image_digest=RELAY_MANIFEST,
        arena_relay_source_sha256=hashlib.sha256(
            Path(B.__file__).with_name(
                "arc_agi3_arena_volume_relay.py"
            ).read_bytes()
        ).hexdigest(),
    )


def test_arena_volume_lifecycle_is_identity_bound_and_proves_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    identity = B.AttemptIdentity.create(game="wa30", target_level=9)
    runner = _ArenaVolumeDockerDouble(identity)
    backend = B.DockerContainerBackend(runner)
    lifecycle = B.DockerArenaVolumeLifecycle(backend)
    attachment = _RelayAttachmentDouble()
    monkeypatch.setattr(
        lifecycle,
        "_start_attachment",
        lambda **_kwargs: attachment,
    )
    rpc_manager = _system_private_test_root(".a3vr_")
    rpc = rpc_manager.__enter__()
    receipts = tmp_path / "receipts"
    receipts.mkdir(mode=0o700)
    rpc.chmod(0o700)
    socket_path = rpc / "arena.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    try:
        runtime = lifecycle.prepare(
            identity=identity,
            transport=_arena_volume_transport(),
            arena_socket=socket_path,
            receipt_root=receipts,
        )
        assert runtime.evidence.volume_name == runner.volume_name
        assert runtime.evidence.relay_container_id == RELAY_CONTAINER_ID
        relay_create = next(
            command
            for command in runner.commands
            if command[:3] == ("docker", "container", "create")
        )
        assert "none" == relay_create[
            relay_create.index("--network") + 1
        ]
        assert (
            f"type=volume,src={runner.volume_name},"
            f"dst={B.PROPOSER_RPC_ROOT_DESTINATION},volume-nocopy"
        ) in relay_create
        proof = lifecycle.teardown(
            identity=identity,
            runtime=runtime,
            receipt_root=receipts,
            containment_fault=False,
        )
    finally:
        listener.close()
        rpc_manager.__exit__(None, None, None)
    assert attachment.finished is True
    assert proof.attachment_status == "CLEAN_EOF"
    assert proof.relay_inspect_absent is True
    assert proof.volume_inspect_absent is True
    assert runner.relay_removed is True
    assert runner.volume_created is False


def test_arena_volume_lifecycle_rejects_writable_proposer_mount_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    identity = B.AttemptIdentity.create(game="wa30", target_level=9)
    runner = _ArenaVolumeDockerDouble(
        identity,
        relay_mutator=lambda value: value["Mounts"][0].__setitem__(
            "RW", False
        ),
    )
    lifecycle = B.DockerArenaVolumeLifecycle(
        B.DockerContainerBackend(runner)
    )
    monkeypatch.setattr(
        lifecycle,
        "_start_attachment",
        lambda **_kwargs: pytest.fail(
            "invalid relay must not attach"
        ),
    )
    rpc_manager = _system_private_test_root(".a3vr_")
    rpc = rpc_manager.__enter__()
    receipts = tmp_path / "receipts"
    receipts.mkdir(mode=0o700)
    rpc.chmod(0o700)
    socket_path = rpc / "arena.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    try:
        with pytest.raises(
            B.ContainerContractError,
            match="writable named-volume mount",
        ):
            lifecycle.prepare(
                identity=identity,
                transport=_arena_volume_transport(),
                arena_socket=socket_path,
                receipt_root=receipts,
            )
    finally:
        listener.close()
        rpc_manager.__exit__(None, None, None)
    assert runner.relay_removed is True
    assert runner.volume_created is False


@pytest.fixture
def attempt_spec(tmp_path: Path, request: pytest.FixtureRequest):
    parent = tmp_path / "parent"
    export = tmp_path / "export"
    parent.mkdir(mode=0o700)
    export.mkdir(mode=0o700)
    (parent / "worker.py").write_text(
        "print('isolated worker')\n", encoding="utf-8"
    )
    # Keep AF_UNIX below macOS's short path limit and inside the test sandbox.
    rpc = _request_private_test_root(
        request, prefix=".a3cb_rpc_"
    )
    rpc.chmod(0o700)
    socket_path = rpc / "arena.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.close()
    socket_path.chmod(0o600)
    token_file = rpc / "token"
    token_file.write_text("t" * 64, encoding="ascii")
    token_file.chmod(0o600)
    spec = B.AttemptSpec(
        identity=B.AttemptIdentity.create(game="wa30", target_level=9),
        image_reference=IMAGE_REFERENCE,
        parent_input=parent,
        export_root=export,
        arena_socket=socket_path,
        arena_token_file=token_file,
        command=B.expected_worker_command(),
        resource_limits=B.ResourceLimits(
            cpus=1.5,
            memory_bytes=768 * 1024 * 1024,
            pids=96,
            tmpfs_bytes=64 * 1024 * 1024,
        ),
        soft_allocation_seconds=90 * 60,
    )
    return spec


def _build(
    spec: B.AttemptSpec,
    **runner_kwargs: Any,
) -> tuple[B.DockerContainerBackend, FakeDockerRunner, B.LaunchAttestation]:
    runner = FakeDockerRunner(spec, **runner_kwargs)
    backend = B.DockerContainerBackend(runner)
    attestation = backend.build_launch_attestation(spec)
    return backend, runner, attestation


def test_identity_generation_is_unique_and_canonical():
    first = B.AttemptIdentity.create(game="wa30", target_level=9)
    second = B.AttemptIdentity.create(
        game="wa30",
        target_level=9,
        generation_id=first.generation_id,
    )
    assert len(
        {first.campaign_id, first.generation_id, first.attempt_id}
    ) == 3
    assert second.generation_id == first.generation_id
    assert second.campaign_id != first.campaign_id
    assert second.attempt_id != first.attempt_id
    assert first.generation_id == second.generation_id
    assert first.attempt_id != second.attempt_id
    assert first.generation_id != first.attempt_id

    with pytest.raises(B.ContainerContractError, match="UUIDv4"):
        dataclasses.replace(first, attempt_id="not-a-uuid").validate()
    with pytest.raises(B.ContainerContractError, match="positive integer"):
        dataclasses.replace(first, target_level=True).validate()


@pytest.mark.parametrize(
    "reference",
    [
        "registry.example/gkm/arc-worker:latest",
        "registry.example/gkm/arc-worker",
        "registry.example/gkm/arc-worker@sha256:abcd",
        "registry.example/gkm/arc-worker@sha256:" + "A" * 64,
        "registry.example/gkm/arc-worker@sha256:" + "a" * 63,
        "registry.example/gkm/arc-worker@sha256:" + "a" * 64 + " trailing",
    ],
)
def test_mutable_or_malformed_image_reference_is_rejected(reference: str):
    with pytest.raises(B.ContainerContractError, match="repository@sha256"):
        B.parse_digest_reference(reference)


def test_well_shaped_fake_digest_not_observed_in_repo_digests_is_rejected(
    attempt_spec: B.AttemptSpec,
):
    runner = FakeDockerRunner(
        attempt_spec,
        image_records=[_image_record(manifest=MANIFEST_B)],
    )
    backend = B.DockerContainerBackend(runner)
    with pytest.raises(
        B.ContainerContractError, match="not present in observed RepoDigests"
    ):
        backend.build_launch_attestation(attempt_spec)
    assert not any(command[2] == "create" for command in runner.commands)


def test_local_tag_only_image_without_repo_digest_is_not_launchable(
    attempt_spec: B.AttemptSpec,
):
    local_only = _image_record()
    local_only["RepoDigests"] = []
    runner = FakeDockerRunner(
        attempt_spec,
        image_records=[local_only],
    )
    with pytest.raises(
        B.ContainerContractError, match="repository-digest evidence"
    ):
        B.DockerContainerBackend(runner).build_launch_attestation(
            attempt_spec
        )
    assert not any(command[2] == "create" for command in runner.commands)


def test_command_runner_cannot_return_evidence_for_a_different_argv(
    attempt_spec: B.AttemptSpec,
):
    class LyingRunner:
        def run(self, argv, *, timeout_seconds=None):
            return B.CommandResult(
                argv=("docker", "image", "inspect", "different@sha256:" + "0" * 64),
                returncode=0,
                stdout=json.dumps([_image_record()]),
            )

    with pytest.raises(B.ContainerContractError, match="bind its result"):
        B.DockerContainerBackend(LyingRunner()).build_launch_attestation(
            attempt_spec
        )


def test_production_command_runner_does_not_inherit_docker_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_ctl_"
    )
    root.chmod(0o700)
    config = root / "config"
    config.mkdir(mode=0o700)
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    captured: dict[str, Any] = {}
    monkeypatch.setenv("DOCKER_HOST", "tcp://attacker.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "tainted-context")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid")
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
    )
    fake_process = SimpleNamespace(
        returncode=0,
        pid=os.getpid(),
        communicate=lambda timeout=None: ("ok", ""),
        poll=lambda: 0,
    )

    def fake_spawn(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return (
            fake_process,
            B._ManagedChildInvocation(
                invocation_id=str(uuid.uuid4()),
                root=root,
                intent_sha256="1" * 64,
                active_sha256="2" * 64,
            ),
        )

    monkeypatch.setattr(runner, "_spawn_managed", fake_spawn)
    monkeypatch.setattr(
        runner, "_finish_managed_invocation", lambda *args, **kwargs: None
    )
    result = runner.run(("/usr/bin/true",), timeout_seconds=1)
    assert result.stdout == "ok"
    assert captured["invocation_kind"] == "bounded_command"
    assert runner._environment == {
        "DOCKER_CONFIG": str(config),
        "DOCKER_HOST": f"unix://{socket_path}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    }
    assert "DOCKER_CONTEXT" not in runner._environment
    assert "HTTPS_PROXY" not in runner._environment
    with pytest.raises(B.ContainerContractError, match="absolute path"):
        B.DockerContainerBackend(runner, docker_binary="docker")


def test_attached_stream_runner_captures_exact_bounded_host_bytes(
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_stream_"
    )
    root.chmod(0o700)
    config = root / "config"
    config.mkdir(mode=0o700)
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
    )
    command = (
        str(Path(sys.executable).resolve()),
        "-c",
        "import sys;sys.stdout.write('out');sys.stderr.write('err')",
    )
    observed = runner.run_attached_stream(
        command,
        timeout_seconds=5,
        stdout_path=evidence / "stdout",
        stderr_path=evidence / "stderr",
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert observed.returncode == 0
    assert observed.timed_out is False
    assert observed.output_overflow is False
    assert observed.stdout_bytes == 3
    assert observed.stderr_bytes == 3
    assert Path(observed.stdout_path).read_bytes() == b"out"
    assert Path(observed.stderr_path).read_bytes() == b"err"
    assert Path(observed.stdout_path).stat().st_mode & 0o777 == 0o400
    assert Path(observed.stderr_path).stat().st_mode & 0o777 == 0o400


def test_attached_stream_runner_kills_on_overflow_or_deadline(
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_stream_"
    )
    root.chmod(0o700)
    config = root / "config"
    config.mkdir(mode=0o700)
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
    )
    executable = str(Path(sys.executable).resolve())

    overflow_root = root / "overflow"
    overflow_root.mkdir(mode=0o700)
    overflow = runner.run_attached_stream(
        (executable, "-c", "print('x' * 10000)"),
        timeout_seconds=5,
        stdout_path=overflow_root / "stdout",
        stderr_path=overflow_root / "stderr",
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert overflow.output_overflow is True
    assert overflow.stdout_truncated is True
    assert overflow.stdout_bytes == 32

    timeout_root = root / "timeout"
    timeout_root.mkdir(mode=0o700)
    timed = runner.run_attached_stream(
        (executable, "-c", "import time;time.sleep(10)"),
        timeout_seconds=1,
        stdout_path=timeout_root / "stdout",
        stderr_path=timeout_root / "stderr",
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert timed.timed_out is True
    assert timed.finished_monotonic - timed.started_monotonic < 5


def test_managed_child_restart_recovery_terminates_multiple_exact_processes(
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_stream_"
    )
    config = root / "config"
    config.mkdir(mode=0o700)
    ledger = root / "child-ledger"
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
        invocation_ledger_root=ledger,
    )
    executable = str(Path(sys.executable).resolve())
    processes = [
        runner._spawn_managed(
            (executable, "-c", "import time;time.sleep(60)"),
            invocation_kind="test_crash_mid_call",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )[0]
        for _ in range(2)
    ]
    child_pids = [process.pid for process in processes]

    recovering_runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
        invocation_ledger_root=ledger,
    )
    recovered = recovering_runner.recover_stale_invocations()

    assert len(recovered) == 0  # constructor performed the recovery
    assert all(
        B._process_start_token(pid) is None for pid in child_pids
    )
    cleanups = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in ledger.glob("*/cleanup.json")
    ]
    assert len(cleanups) == 2
    assert all(
        item["outcome"] == "EXACT_CHILD_TERMINATED"
        and item["identity_matched"] is True
        and item["status"] == "CLEAN"
        for item in cleanups
    )
    audit = recovering_runner.audit_invocation_ledger()
    assert audit["kind"] == (
        "arc_agi3_managed_host_child_ledger_audit"
    )
    assert audit["invocation_count"] == 2
    assert audit["status_counts"] == {
        "PENDING": 0,
        "ACTIVE": 0,
        "TERMINAL": 0,
        "CLEAN": 2,
    }
    assert audit["startup_recovered_count"] == 2
    assert {
        item["invocation_id"]
        for item in audit["startup_recovery"]
    } == {
        item["invocation_id"] for item in cleanups
    }
    unsigned_audit = dict(audit)
    observed_authentication = unsigned_audit.pop(
        "authentication_sha256"
    )
    assert observed_authentication == (
        B._managed_process_authentication(
            recovering_runner._invocation_authentication_key,
            unsigned_audit,
        )
    )


def test_managed_child_pid_reuse_mismatch_is_never_killed(
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_stream_"
    )
    config = root / "config"
    config.mkdir(mode=0o700)
    ledger = root / "child-ledger"
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
        invocation_ledger_root=ledger,
    )
    command = (str(Path(sys.executable).resolve()), "-c", "pass")
    invocation = runner._begin_managed_invocation(
        command, invocation_kind="test_pid_reuse"
    )
    active_body = {
        "schema": 1,
        "kind": "arc_agi3_managed_host_child_active",
        "invocation_id": invocation.invocation_id,
        "intent_sha256": invocation.intent_sha256,
        "argv_sha256": B._argv_sha256(command),
        "child_pid": os.getpid(),
        "child_start_token": "synthetic-reused-incarnation",
        "child_pgid": os.getpgid(0),
        "child_sid": os.getsid(0),
        "operator_sid": os.getsid(0),
        "activated_at": time.time(),
        "status": "ACTIVE",
    }
    B._write_private_json_new(
        invocation.root / "active.json",
        runner._authenticated_invocation_document(active_body),
    )

    recovered = runner.recover_stale_invocations()

    assert len(recovered) == 1
    cleanup = json.loads(
        Path(recovered[0]).read_text(encoding="utf-8")
    )
    assert cleanup["outcome"] == "PID_REUSED_NOT_KILLED"
    assert cleanup["identity_matched"] is False
    assert B._process_start_token(os.getpid()) is not None


def test_managed_child_ledger_audit_reopens_terminal_and_rejects_drift(
    request: pytest.FixtureRequest,
):
    root = _request_private_test_root(
        request, prefix=".a3cb_stream_"
    )
    config = root / "config"
    config.mkdir(mode=0o700)
    ledger = root / "child-ledger"
    socket_path = root / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    request.addfinalizer(listener.close)
    runner = B.SubprocessCommandRunner(
        docker_socket=socket_path,
        docker_config=config,
        invocation_ledger_root=ledger,
    )
    executable = str(Path(sys.executable).resolve())

    result = runner.run(
        (executable, "-c", "print('bounded')"),
        timeout_seconds=5,
    )
    assert result.returncode == 0
    interactive = runner.spawn_interactive(
        (executable, "-c", "print('interactive')")
    )
    stdout, stderr = interactive.communicate(timeout=5)
    assert interactive.returncode == 0
    assert stdout == b"interactive\n"
    assert stderr == b""

    audit = runner.audit_invocation_ledger()
    assert audit["invocation_count"] == 2
    assert audit["status_counts"] == {
        "PENDING": 0,
        "ACTIVE": 0,
        "TERMINAL": 2,
        "CLEAN": 0,
    }
    assert all(
        item["ledger_status"] == "TERMINAL"
        and item["terminal_sha256"] is not None
        for item in audit["records"]
    )

    invocation_root = next(
        path
        for path in ledger.iterdir()
        if path.is_dir()
    )
    (invocation_root / "unexpected").write_text(
        "drift", encoding="utf-8"
    )
    with pytest.raises(
        B.ContainerContractError,
        match="unexpected receipts",
    ):
        runner.audit_invocation_ledger()


def test_image_must_bind_exact_current_rpc_and_worker_sources(
    attempt_spec: B.AttemptSpec,
):
    for label in B.trusted_worker_hashes():
        missing = _image_record()
        missing["Config"]["Labels"].pop(label)
        runner = FakeDockerRunner(attempt_spec, image_records=[missing])
        with pytest.raises(
            B.ContainerContractError, match="pinned worker controls"
        ):
            B.DockerContainerBackend(runner).build_launch_attestation(
                attempt_spec
            )
        assert not any(
            command[2] == "create" for command in runner.commands
        )

        stale = _image_record()
        stale["Config"]["Labels"][label] = "0" * 64
        runner = FakeDockerRunner(attempt_spec, image_records=[stale])
        with pytest.raises(
            B.ContainerContractError, match="pinned worker controls"
        ):
            B.DockerContainerBackend(runner).build_launch_attestation(
                attempt_spec
            )
        assert not any(
            command[2] == "create" for command in runner.commands
        )


def test_image_or_container_auth_environment_is_rejected(
    attempt_spec: B.AttemptSpec,
):
    image_with_auth = _image_record()
    image_with_auth["Config"]["Env"].append(
        "OPENAI_API_KEY=must-never-enter-the-image"
    )
    runner = FakeDockerRunner(
        attempt_spec, image_records=[image_with_auth]
    )
    with pytest.raises(
        B.ContainerContractError, match="unapproved environment"
    ):
        B.DockerContainerBackend(runner).build_launch_attestation(
            attempt_spec
        )
    assert not any(command[2] == "create" for command in runner.commands)

    def inject_auth(record: dict[str, Any]) -> None:
        record["Config"]["Env"].append(
            "CHATGPT_TOKEN=must-never-enter-the-container"
        )

    runner = FakeDockerRunner(
        attempt_spec, container_mutator=inject_auth
    )
    with pytest.raises(
        B.ContainerContractError, match="unapproved environment"
    ):
        B.DockerContainerBackend(runner).build_launch_attestation(
            attempt_spec
        )
    assert runner.removed is True


def test_create_command_has_exact_fail_closed_isolation(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    create = next(command for command in runner.commands if command[2] == "create")

    assert create[:3] == ("docker", "container", "create")
    assert "--read-only" in create
    assert create[create.index("--network") + 1] == "none"
    assert "--pid" not in create
    assert "--uts" not in create
    assert create[create.index("--cgroupns") + 1] == "private"
    assert create[create.index("--ipc") + 1] == "private"
    assert create[create.index("--cap-drop") + 1] == "ALL"
    assert (
        create[create.index("--security-opt") + 1]
        == "no-new-privileges=true"
    )
    assert create[create.index("--user") + 1] == (
        f"{attempt_spec.export_root.stat().st_uid}:"
        f"{attempt_spec.export_root.stat().st_gid}"
    )
    assert create[create.index("--pids-limit") + 1] == "96"
    assert create[create.index("--restart") + 1] == "no"
    assert "--no-healthcheck" in create
    assert create[create.index("--log-driver") + 1] == "local"
    log_options = [
        create[index + 1]
        for index, value in enumerate(create)
        if value == "--log-opt"
    ]
    assert log_options == ["max-size=4m", "max-file=1"]
    assert create[create.index("--pull") + 1] == "never"
    assert create[create.index("--memory") + 1] == str(768 * 1024 * 1024)
    assert create[create.index("--memory-swap") + 1] == str(768 * 1024 * 1024)
    assert create[create.index("--entrypoint") + 1] == B.PYTHON_ENTRYPOINT
    assert (
        f"{B.LABEL_CAMPAIGN}={attempt_spec.identity.campaign_id}" in create
    )
    assert (
        f"ARC_AGI3_CAMPAIGN_ID={attempt_spec.identity.campaign_id}" in create
    )
    assert create[-len(attempt_spec.command) - 1] == IMAGE_REFERENCE
    assert create[-len(attempt_spec.command) :] == attempt_spec.command
    mounts = [
        create[index + 1]
        for index, value in enumerate(create)
        if value == "--mount"
    ]
    assert len(mounts) == 4
    assert "dst=/arc/input,readonly" in mounts[0]
    assert "dst=/arc/export,bind-propagation=rprivate" in mounts[1]
    assert "readonly" not in mounts[1]
    assert f"dst={B.RPC_SOCKET_DESTINATION},readonly" in mounts[2]
    assert f"dst={B.RPC_TOKEN_DESTINATION},readonly" in mounts[3]
    assert attempt_spec.command == (
        "-I",
        "-m",
        "arc_agi3_container_worker",
        "--socket=/run/arc-agi3/arena.sock",
        "--token-file=/run/arc-agi3/token",
        "--solve=/arc/input/solve.py",
        "--outcome=/arc/export/worker_outcome.json",
    )
    assert all("docker.sock" not in value for value in create)
    assert attestation.create_argv_sha256 == B._argv_sha256(create)


def test_attestation_contains_only_observation_derived_isolation_evidence(
    attempt_spec: B.AttemptSpec,
):
    _, _, attestation = _build(attempt_spec)
    assert {fact.name for fact in attestation.evidence} == set(
        B.ISOLATION_EVIDENCE_NAMES
    )
    assert all(
        fact.source == "docker-container-inspect"
        and fact.observation_sha256
        == attestation.container_observation_sha256
        for fact in attestation.evidence
    )
    document = attestation.as_dict()
    assert document["campaign_id"] == attempt_spec.identity.campaign_id
    assert set(document["isolation_evidence"]) == set(
        B.ISOLATION_EVIDENCE_NAMES
    )
    assert not any(
        isinstance(value, bool)
        for evidence in document["isolation_evidence"].values()
        for value in evidence.values()
    )
    assert document["image"]["requested_reference"] == IMAGE_REFERENCE
    assert document["image"]["manifest_digest"] == MANIFEST_A
    assert document["image"]["image_id"] == IMAGE_ID_A


def test_digest_drift_between_inspection_and_create_fails_before_create(
    attempt_spec: B.AttemptSpec,
):
    runner = FakeDockerRunner(
        attempt_spec,
        image_records=[
            _image_record(image_id=IMAGE_ID_A),
            _image_record(image_id=IMAGE_ID_B),
        ],
    )
    backend = B.DockerContainerBackend(runner)
    with pytest.raises(B.ContainerContractError, match="identity drifted"):
        backend.build_launch_attestation(attempt_spec)
    assert not any(command[2] == "create" for command in runner.commands)


def test_digest_drift_after_attestation_blocks_start(
    attempt_spec: B.AttemptSpec,
):
    runner = FakeDockerRunner(
        attempt_spec,
        image_records=[
            _image_record(image_id=IMAGE_ID_A),
            _image_record(image_id=IMAGE_ID_A),
            _image_record(image_id=IMAGE_ID_B),
        ],
    )
    backend = B.DockerContainerBackend(runner)
    attestation = backend.build_launch_attestation(attempt_spec)
    with pytest.raises(B.ContainerContractError, match="identity drifted"):
        backend.start_attested(attestation, attempt_spec)
    assert not any(command[2] == "start" for command in runner.commands)


def test_container_created_with_mutable_tag_is_rejected_and_removed(
    attempt_spec: B.AttemptSpec,
):
    def mutate(record: dict[str, Any]) -> None:
        record["Config"]["Image"] = "registry.example/gkm/arc-worker:latest"

    runner = FakeDockerRunner(attempt_spec, container_mutator=mutate)
    backend = B.DockerContainerBackend(runner)
    with pytest.raises(B.ContainerContractError, match="repository@sha256"):
        backend.build_launch_attestation(attempt_spec)
    assert runner.removed
    assert any(command[2] == "rm" for command in runner.commands)


def test_malformed_create_ack_recovers_exact_labelled_container(
    attempt_spec: B.AttemptSpec,
):
    class MalformedCreateAckRunner(FakeDockerRunner):
        def run(self, argv, *, timeout_seconds=None):
            result = super().run(argv, timeout_seconds=timeout_seconds)
            if tuple(argv)[:3] == ("docker", "container", "create"):
                return B.CommandResult(
                    argv=tuple(argv),
                    returncode=0,
                    stdout="truncated-id\n",
                )
            return result

    runner = MalformedCreateAckRunner(attempt_spec)
    with pytest.raises(
        B.ContainerContractError, match="identity-labelled container was removed"
    ):
        B.DockerContainerBackend(runner).build_launch_attestation(
            attempt_spec
        )
    assert runner.removed is True
    assert any(
        command[:3] == ("docker", "container", "ls")
        for command in runner.commands
    )
    assert any(
        command[:3] == ("docker", "container", "rm")
        for command in runner.commands
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["HostConfig"].pop("PidMode"),
            "PidMode is absent",
        ),
        (
            lambda record: record["HostConfig"].update(CgroupnsMode="host"),
            "cgroup namespace",
        ),
        (
            lambda record: record["HostConfig"]["LogConfig"].update(
                Config={"max-size": "unlimited"}
            ),
            "log retention",
        ),
        (
            lambda record: record["Config"].update(
                Healthcheck={"Test": ["CMD", "sh", "-c", "true"]}
            ),
            "healthcheck",
        ),
        (
            lambda record: record["HostConfig"].update(NetworkMode="bridge"),
            "network is not disabled",
        ),
        (
            lambda record: record["HostConfig"].update(ReadonlyRootfs=False),
            "root filesystem is writable",
        ),
        (
            lambda record: record["HostConfig"].update(CapDrop=[]),
            "capabilities are not exactly dropped",
        ),
        (
            lambda record: record["HostConfig"].update(SecurityOpt=[]),
            "no-new-privileges",
        ),
        (
            lambda record: record["HostConfig"].update(PidsLimit=0),
            "PID bound differs",
        ),
        (
            lambda record: record["Mounts"].append(
                {
                    "Type": "bind",
                    "Source": "/var/run/docker.sock",
                    "Destination": "/var/run/docker.sock",
                    "RW": True,
                    "Propagation": "rprivate",
                }
            ),
            "exact role",
        ),
        (
            lambda record: record["NetworkSettings"].update(
                Networks={"bridge": {}}
            ),
            "attached networks",
        ),
    ],
)
def test_missing_or_failed_isolation_evidence_fails_closed_and_cleans_up(
    attempt_spec: B.AttemptSpec,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    runner = FakeDockerRunner(attempt_spec, container_mutator=mutate)
    backend = B.DockerContainerBackend(runner)
    with pytest.raises(B.ContainerContractError, match=message):
        backend.build_launch_attestation(attempt_spec)
    assert runner.removed
    assert not any(command[2] == "start" for command in runner.commands)


def test_symlinked_mount_root_is_rejected_before_docker(
    attempt_spec: B.AttemptSpec,
    tmp_path: Path,
):
    link = tmp_path / "parent-link"
    link.symlink_to(attempt_spec.parent_input, target_is_directory=True)
    linked_spec = dataclasses.replace(attempt_spec, parent_input=link)
    runner = FakeDockerRunner(linked_spec)
    backend = B.DockerContainerBackend(runner)
    with pytest.raises(B.ContainerContractError, match="symlinked"):
        backend.build_launch_attestation(linked_spec)
    assert runner.commands == []


def test_rpc_socket_and_token_are_real_private_unaliased_endpoints(
    attempt_spec: B.AttemptSpec,
    tmp_path: Path,
):
    ordinary = tmp_path / "not-a-socket"
    ordinary.write_text("x", encoding="utf-8")
    ordinary.chmod(0o600)
    wrong_socket = dataclasses.replace(attempt_spec, arena_socket=ordinary)
    runner = FakeDockerRunner(wrong_socket)
    with pytest.raises(B.ContainerContractError, match="Unix socket"):
        B.DockerContainerBackend(runner).build_launch_attestation(wrong_socket)
    assert runner.commands == []

    actual = tmp_path / "actual-token"
    actual.write_text("x" * 64, encoding="ascii")
    actual.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(actual)
    linked_token = dataclasses.replace(attempt_spec, arena_token_file=link)
    runner = FakeDockerRunner(linked_token)
    with pytest.raises(B.ContainerContractError, match="symlinked"):
        B.DockerContainerBackend(runner).build_launch_attestation(linked_token)
    assert runner.commands == []

    attempt_spec.arena_token_file.chmod(0o644)
    runner = FakeDockerRunner(attempt_spec)
    with pytest.raises(B.ContainerContractError, match="group/other"):
        B.DockerContainerBackend(runner).build_launch_attestation(attempt_spec)
    assert runner.commands == []


def test_rpc_endpoint_identity_or_token_content_drift_blocks_start(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    token = attempt_spec.arena_token_file
    token.unlink()
    token.write_text("u" * 64, encoding="ascii")
    token.chmod(0o600)
    with pytest.raises(B.ContainerContractError, match="changed after attestation"):
        backend.start_attested(attestation, attempt_spec)
    assert not any(command[2] == "start" for command in runner.commands)


def test_arbitrary_entrypoint_command_is_rejected_before_docker(
    attempt_spec: B.AttemptSpec,
):
    unsafe = dataclasses.replace(
        attempt_spec,
        command=("-I", "-c", "print('bypass fixed worker')"),
    )
    runner = FakeDockerRunner(unsafe)
    with pytest.raises(B.ContainerContractError, match="exact pinned"):
        B.DockerContainerBackend(runner).build_launch_attestation(unsafe)
    assert runner.commands == []


def test_same_or_nested_mount_roots_are_rejected_before_docker(
    attempt_spec: B.AttemptSpec,
):
    same = dataclasses.replace(
        attempt_spec,
        parent_input=attempt_spec.export_root,
    )
    runner = FakeDockerRunner(same)
    with pytest.raises(B.ContainerContractError, match="aliases"):
        B.DockerContainerBackend(runner).build_launch_attestation(same)
    assert runner.commands == []

    nested = attempt_spec.parent_input / "nested-export"
    nested.mkdir(mode=0o700)
    nested_spec = dataclasses.replace(attempt_spec, export_root=nested)
    nested_runner = FakeDockerRunner(nested_spec)
    with pytest.raises(B.ContainerContractError, match="must not overlap"):
        B.DockerContainerBackend(nested_runner).build_launch_attestation(
            nested_spec
        )
    assert nested_runner.commands == []


def test_hardlinked_parent_file_and_nonempty_export_are_rejected(
    attempt_spec: B.AttemptSpec,
):
    hardlink = attempt_spec.parent_input / "worker-alias.py"
    os.link(attempt_spec.parent_input / "worker.py", hardlink)
    runner = FakeDockerRunner(attempt_spec)
    with pytest.raises(B.ContainerContractError, match="aliased regular file"):
        B.DockerContainerBackend(runner).build_launch_attestation(attempt_spec)
    assert runner.commands == []
    hardlink.unlink()

    (attempt_spec.export_root / "stale.txt").write_text(
        "old attempt", encoding="utf-8"
    )
    with pytest.raises(B.ContainerContractError, match="empty output-only"):
        B.DockerContainerBackend(runner).build_launch_attestation(attempt_spec)


def test_parent_or_export_change_after_attestation_blocks_start(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    (attempt_spec.parent_input / "late.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(B.ContainerContractError, match="changed after attestation"):
        backend.start_attested(attestation, attempt_spec)
    assert not any(command[2] == "start" for command in runner.commands)

    (attempt_spec.parent_input / "late.py").unlink()
    # A fresh builder is needed because the first created container remains
    # deliberately unstarted; use a new identity/root for the export case.
    second_spec = dataclasses.replace(
        attempt_spec,
        identity=B.AttemptIdentity.create(game="wa30", target_level=9),
    )
    second_backend, second_runner, second = _build(second_spec)
    (second_spec.export_root / "foreign.txt").write_text("x", encoding="utf-8")
    with pytest.raises(B.ContainerContractError, match="empty output-only"):
        second_backend.start_attested(second, second_spec)
    assert not any(command[2] == "start" for command in second_runner.commands)


def test_tampered_or_incomplete_attestation_cannot_authorize_start(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    tampered_bytes = dataclasses.replace(
        attestation,
        document_bytes=attestation.document_bytes + b" ",
    )
    with pytest.raises(B.ContainerContractError, match="bytes were modified"):
        backend.start_attested(tampered_bytes, attempt_spec)

    missing_fact = dataclasses.replace(
        attestation,
        evidence=attestation.evidence[:-1],
    )
    with pytest.raises(B.ContainerContractError, match="does not bind"):
        backend.start_attested(missing_fact, attempt_spec)
    assert not any(command[2] == "start" for command in runner.commands)


def test_soft_allocation_only_enters_draining_and_never_calls_docker(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    before = list(runner.commands)
    assert backend.observe_soft_allocation(
        attestation,
        elapsed_seconds=89 * 60,
        proposer_active=True,
    ) == B.SoftAllocationDecision(
        phase="PROPOSING",
        launch_new_turn=False,
    )
    for elapsed in (90 * 60, 180 * 60, 24 * 60 * 60):
        assert backend.observe_soft_allocation(
            attestation,
            elapsed_seconds=elapsed,
            proposer_active=True,
        ) == B.SoftAllocationDecision(
            phase="DRAINING",
            launch_new_turn=False,
        )
    assert runner.commands == before
    with pytest.raises(B.ContainerContractError, match="TeardownCause"):
        backend.teardown(attestation, cause="soft_allocation")  # type: ignore[arg-type]
    assert runner.commands == before


def test_attested_start_rechecks_image_mounts_and_container_then_runs(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    running = backend.start_attested(attestation, attempt_spec)
    assert running.attestation is attestation
    starts = [command for command in runner.commands if command[2] == "start"]
    assert starts == [("docker", "container", "start", CONTAINER_ID)]
    assert runner.running


def test_terminal_log_capture_retains_bytes_beyond_diagnostic_excerpt(
    attempt_spec: B.AttemptSpec,
):
    marker = "FORBIDDEN_TAINT_MARKER_AFTER_EXCERPT"
    stdout = "x" * (B.MAX_CAPTURED_LOG_CHARS + 1) + marker
    backend, runner, attestation = _build(
        attempt_spec, log_stdout=stdout, log_stderr="diagnostic stderr"
    )
    backend.start_attested(attestation, attempt_spec)
    runner.running = False

    observed = backend.collect_terminal_logs(attestation, attempt_spec)

    assert observed.stdout.endswith(marker)
    assert observed.stdout_bytes == len(stdout.encode("utf-8"))
    assert observed.stdout_sha256 == hashlib.sha256(
        stdout.encode("utf-8")
    ).hexdigest()
    assert observed.stderr == "diagnostic stderr"


def test_terminal_log_capture_fails_closed_above_full_stream_bound(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(
        attempt_spec,
        log_stdout="x" * (B.MAX_CONTAINER_STREAM_BYTES + 1),
    )
    backend.start_attested(attestation, attempt_spec)
    runner.running = False

    with pytest.raises(B.ContainerContractError, match="stream exceeds"):
        backend.collect_terminal_logs(attestation, attempt_spec)


def test_teardown_is_exactly_scoped_and_proves_no_descendants(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)
    running = backend.start_attested(attestation, attempt_spec)
    proof = backend.teardown(
        running,
        cause=B.TeardownCause.CONTAINMENT_FAULT,
        graceful_seconds=7,
    )
    assert proof.no_descendants
    assert proof.container_inspect_absent
    assert proof.container_top_absent
    assert proof.identity_label_query_empty
    assert proof.campaign_id == attempt_spec.identity.campaign_id
    assert proof.observed_container_processes_before == (4242, 4243)
    stop = next(command for command in runner.commands if command[2] == "stop")
    remove = next(command for command in runner.commands if command[2] == "rm")
    assert stop == (
        "docker",
        "container",
        "stop",
        "--time",
        "7",
        CONTAINER_ID,
    )
    assert remove == (
        "docker",
        "container",
        "rm",
        "--force",
        "--volumes",
        CONTAINER_ID,
    )
    assert not any(command[2] == "kill" for command in runner.commands)
    identity_query = next(
        command for command in runner.commands if command[2] == "ls"
    )
    assert (
        f"label={B.LABEL_CAMPAIGN}={attempt_spec.identity.campaign_id}"
        in identity_query
    )
    assert (
        f"label={B.LABEL_GENERATION}={attempt_spec.identity.generation_id}"
        in identity_query
    )
    assert (
        f"label={B.LABEL_ATTEMPT}={attempt_spec.identity.attempt_id}"
        in identity_query
    )


def test_teardown_fails_if_identity_query_finds_a_surviving_container(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(
        attempt_spec, label_query_output="deadbeef\n"
    )
    running = backend.start_attested(attestation, attempt_spec)
    with pytest.raises(B.ContainerContractError, match="could not prove"):
        backend.teardown(
            running,
            cause=B.TeardownCause.EXPLICIT_SUPERVISOR_SHUTDOWN,
        )


def test_wrong_container_image_id_blocks_start_and_teardown(
    attempt_spec: B.AttemptSpec,
):
    backend, runner, attestation = _build(attempt_spec)

    def mutate(record: dict[str, Any]) -> None:
        record["Image"] = IMAGE_ID_B

    runner.container_mutator = mutate
    with pytest.raises(B.ContainerContractError, match="image id differs"):
        backend.start_attested(attestation, attempt_spec)
    with pytest.raises(B.ContainerContractError, match="drifted image"):
        backend.teardown(
            attestation,
            cause=B.TeardownCause.EXPLICIT_SUPERVISOR_SHUTDOWN,
        )


def test_attestation_is_exclusive_host_owned_file(
    attempt_spec: B.AttemptSpec,
    tmp_path: Path,
):
    backend, _, attestation = _build(attempt_spec)
    output = tmp_path / "attestations"
    output.mkdir(mode=0o700)
    path = output / "launch.json"
    backend.write_attestation(path, attestation)
    assert path.stat().st_nlink == 1
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == attestation.document_bytes
    with pytest.raises(B.ContainerContractError, match="new regular file"):
        backend.write_attestation(path, attestation)

    target = output / "elsewhere"
    target.write_text("do not overwrite", encoding="utf-8")
    link = output / "linked.json"
    link.symlink_to(target)
    with pytest.raises(B.ContainerContractError, match="new regular file"):
        backend.write_attestation(link, attestation)
    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_formal_container_recipe_pins_base_and_nonroot_user():
    recipe = (
        Path(__file__).parent
        / "container"
        / "Containerfile.arc-agi3-contiguous"
    ).read_text(encoding="utf-8")
    from_line = next(
        line for line in recipe.splitlines() if line.startswith("FROM ")
    )
    assert "@sha256:" in from_line
    assert len(from_line.rsplit("@sha256:", 1)[1]) == 64
    assert "USER 65532:65532" in recipe
    assert 'ENTRYPOINT ["/usr/local/bin/python3", "-I"]' in recipe
    assert "arc_agi3_arena_rpc_client.py" in recipe
    assert "arc_agi3_arena_rpc.py" not in recipe
    assert "arc_agi3_container_worker.py" in recipe
    assert "arc_agi3_proposer_worker.py" in recipe
    assert B.LABEL_PROPOSER_WORKER_SHA256 in recipe
    assert B.LABEL_SOURCE_SCHEMA_SHA256 in recipe
    assert B.LABEL_CONTAINER_RECIPE_SHA256 in recipe
    assert B.LABEL_SOLVER_REQUIREMENTS_SHA256 in recipe
    assert "ARG ARC_AGI3_CONTAINER_RECIPE_SHA256\n" in recipe
    assert "numpy-version" not in recipe
    assert "/run/arc-agi3/arena.sock" in recipe
    assert "/run/arc-agi3/token" in recipe
    for label, digest in B.trusted_worker_hashes().items():
        if label == B.LABEL_CONTAINER_RECIPE_SHA256:
            continue
        assert digest in recipe
    assert "sha256sum --check --strict" in recipe
    assert "contiguous campaign" in recipe.lower()


def _runner_attempt_spec(tmp_path: Path):
    import arc_agi3_contiguous_runner as R
    import arc_agi3_contiguous_supervisor as S

    campaign_id = str(uuid.uuid4())
    generation_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    attempts = (tmp_path / "campaign" / "generations").resolve()
    attempts.mkdir(mode=0o700, parents=True)
    os.chmod(attempts.parent, 0o700)
    generation = (attempts / generation_id).resolve()
    generation.mkdir(mode=0o700)
    for name in (
        "input",
        "scratch",
        "output",
        "rpc",
        "bridge",
        "host",
        "state",
    ):
        child = generation / name
        child.mkdir(mode=0o700)
        child.chmod(0o700)
    for name in ("neutral", "app_server_control"):
        child = generation / "host" / name
        child.mkdir(mode=0o700)
        child.chmod(0o700)
    (generation / "host" / "neutral").chmod(0o500)
    codex_home = generation / "state" / "codex_home"
    codex_home.mkdir(mode=0o700)
    codex_home.chmod(0o700)
    generation.chmod(0o700)
    parent = (tmp_path / "parent_checkpoint.json").resolve()
    parent.write_text(
        json.dumps(
            {
                "game": "wa30",
                "reached": 8,
                "total_marginal_C": 8,
                "records": [
                    {"level": level, "marginal_C": 1, "reached": True}
                    for level in range(1, 9)
                ],
                "final_path": [1] * 8,
                "validated": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    parent_sha256 = hashlib.sha256(parent.read_bytes()).hexdigest()
    lane_parent_source = (tmp_path / "lane_parent_source").resolve()
    lane_parent_source.mkdir(mode=0o700)
    parent_source = generation / "input" / "parent_source"
    parent_source.mkdir(mode=0o700)
    source_payloads = {
        "legs.py": "LEGS = ()\n",
        "players.py": "PLAYERS = ()\n",
        "solve.py": "def solve(env):\n    return None\n",
    }
    for name, payload in source_payloads.items():
        (lane_parent_source / name).write_text(
            payload, encoding="utf-8"
        )
        (parent_source / name).write_text(payload, encoding="utf-8")
        (generation / "scratch" / name).write_text(
            payload, encoding="utf-8"
        )
    trusted = (tmp_path / "trusted").resolve()
    trusted.mkdir(mode=0o700)
    codex_binary = trusted / "codex"
    codex_binary.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' 'codex-cli 1.2.3'\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    codex_binary.chmod(0o500)
    codex_launcher = trusted / "codex-launcher"
    codex_launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex_launcher.chmod(0o500)
    package_manifest = trusted / "package.json"
    package_manifest.write_text(
        '{"name":"codex-test","version":"1.2.3"}\n',
        encoding="utf-8",
    )
    package_manifest.chmod(0o400)
    protocol_schema = trusted / "app_server_protocol.json"
    protocol_schema.write_text(
        '{"schema":"test-app-server"}\n', encoding="utf-8"
    )
    protocol_schema.chmod(0o400)
    protocol_bundle = trusted / "app_server_protocol_bundle.json"
    protocol_bundle.write_text(
        '{"bundle":"test-app-server"}\n', encoding="utf-8"
    )
    protocol_bundle.chmod(0o400)
    transport = R.ProposerTransportConfiguration(
        model="gpt-5.6-sol",
        model_provider="openai",
        allow_provider_model_fallback=False,
        reasoning_effort_allowlist=(
            R.EXPECTED_REASONING_EFFORT_ALLOWLIST
        ),
        controller_image_reference=(
            "gkm/arc-controller@sha256:" + "9" * 64
        ),
        controller_image_digest="sha256:" + "9" * 64,
        controller_entrypoint=R.EXPECTED_CONTROLLER_ENTRYPOINT,
        controller_guardian_path=R.EXPECTED_CONTROLLER_ENTRYPOINT[0],
        controller_guardian_sha256="a" * 64,
        controller_user=R.EXPECTED_CONTROLLER_USER,
        controller_egress_policy=R.EXPECTED_CONTROLLER_EGRESS_POLICY,
        controller_egress_proxy_image_reference=(
            "gkm/arc-egress-proxy@sha256:" + "b" * 64
        ),
        controller_egress_proxy_image_digest="sha256:" + "b" * 64,
        controller_egress_policy_sha256="c" * 64,
        controller_cpus=2.0,
        controller_memory_bytes=4 * 1024**3,
        controller_pids=256,
        controller_tmpfs_bytes=512 * 1024**2,
        arena_transport="docker-attach-stdio+named-volume-unix",
        arena_relay_image_reference=(
            "gkm/arc-arena-relay@sha256:" + "d" * 64
        ),
        arena_relay_image_digest="sha256:" + "d" * 64,
        arena_relay_source_sha256=hashlib.sha256(
            Path(B.__file__).with_name(
                "arc_agi3_arena_volume_relay.py"
            ).read_bytes()
        ).hexdigest(),
        codex_launcher_path=str(codex_launcher),
        codex_launcher_sha256=hashlib.sha256(
            codex_launcher.read_bytes()
        ).hexdigest(),
        codex_package_manifest_path=str(package_manifest),
        codex_package_manifest_sha256=hashlib.sha256(
            package_manifest.read_bytes()
        ).hexdigest(),
        codex_binary_path=str(codex_binary),
        codex_binary_sha256=hashlib.sha256(
            codex_binary.read_bytes()
        ).hexdigest(),
        codex_binary_bytes=codex_binary.stat().st_size,
        codex_cli_version="codex-cli 1.2.3",
        app_server_protocol_schema_path=str(protocol_schema),
        app_server_protocol_schema_sha256=hashlib.sha256(
            protocol_schema.read_bytes()
        ).hexdigest(),
        app_server_protocol_schema_bundle_path=str(protocol_bundle),
        app_server_protocol_schema_bundle_sha256=hashlib.sha256(
            protocol_bundle.read_bytes()
        ).hexdigest(),
        controller_preflight_request_allowlist=(
            R.EXPECTED_CONTROLLER_PREFLIGHT_REQUEST_ALLOWLIST
        ),
        controller_preflight_notification_allowlist=(
            R.EXPECTED_CONTROLLER_PREFLIGHT_NOTIFICATION_ALLOWLIST
        ),
        controller_turn_request_allowlist=(
            R.EXPECTED_CONTROLLER_TURN_REQUEST_ALLOWLIST
        ),
        dynamic_tool_namespace=R.EXPECTED_DYNAMIC_TOOL_NAMESPACE,
        dynamic_tool_names=R.EXPECTED_DYNAMIC_TOOL_NAMES,
        bridge_protocol_version=1,
        bridge_operation_allowlist=(
            R.EXPECTED_BRIDGE_OPERATION_ALLOWLIST
        ),
        bridge_exec_allowlist=R.EXPECTED_BRIDGE_EXEC_ALLOWLIST,
        bridge_max_request_bytes=1024 * 1024,
        bridge_max_response_bytes=1024 * 1024,
        bridge_max_file_bytes=8 * 1024 * 1024,
        bridge_max_total_export_bytes=32 * 1024 * 1024,
        bridge_max_processes=16,
        bridge_max_exec_seconds=600,
    )
    frontier_hash = R.frontier_sha256("wa30", 8, parent_sha256)
    (generation / "input" / "checkpoint.json").write_bytes(
        parent.read_bytes()
    )
    frontier_brief = generation / "input" / "frontier_brief.json"
    frontier_brief.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_frontier_brief",
                "campaign_id": campaign_id,
                "generation_id": generation_id,
                "attempt_id": attempt_id,
                "game": "wa30",
                "target_level": 9,
                "authoritative_target": 9,
                "parent_checkpoint_sha256": parent_sha256,
                "frontier_sha256": frontier_hash,
                "parent_action_count": 8,
                "remaining_action_budget": 592,
                "fresh_prefix_required": False,
                "effort": "max",
                "soft_allocation_seconds": 90 * 60,
                    "wip_mode": "exclude",
                    "thread_mode": "new",
                    "supervisory_handoff": None,
                },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    bridge_bounds = {
        "max_request_bytes": transport.bridge_max_request_bytes,
        "max_response_bytes": transport.bridge_max_response_bytes,
        "max_file_bytes": transport.bridge_max_file_bytes,
        "max_total_export_bytes":
            transport.bridge_max_total_export_bytes,
        "max_processes": transport.bridge_max_processes,
        "max_exec_seconds": transport.bridge_max_exec_seconds,
    }
    (generation / "input" / "bridge_policy.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_bridge_policy",
                "campaign_id": campaign_id,
                "generation_id": generation_id,
                "attempt_id": attempt_id,
                "game": "wa30",
                "target_level": 9,
                "frontier_sha256": frontier_hash,
                "parent_checkpoint_sha256": parent_sha256,
                "protocol_version": transport.bridge_protocol_version,
                "operation_allowlist": list(
                    transport.bridge_operation_allowlist
                ),
                "exec_allowlist": list(
                    transport.bridge_exec_allowlist
                ),
                "bounds": bridge_bounds,
                "workspace_root": B.WORKSPACE_DESTINATION,
                "export_root": B.EXPORT_DESTINATION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    bridge_policy = generation / "input" / "bridge_policy.json"
    (generation / "scratch" / "checkpoint.json").write_bytes(
        parent.read_bytes()
    )
    (generation / "scratch" / "frontier_brief.json").write_bytes(
        frontier_brief.read_bytes()
    )
    frontier_brief_sha256 = hashlib.sha256(
        frontier_brief.read_bytes()
    ).hexdigest()
    bridge_policy_sha256 = hashlib.sha256(
        bridge_policy.read_bytes()
    ).hexdigest()
    input_tree_sha256 = S._tree_hash(generation / "input")
    parent_source_tree_sha256 = S._tree_hash(parent_source)
    initial_workspace_tree_sha256 = S._tree_hash(
        generation / "scratch"
    )
    initial_app_server_state_tree_sha256 = S._tree_hash(codex_home)
    receipt = generation / "input_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": R.RUNNER_SCHEMA,
                "campaign_id": campaign_id,
                "generation_id": generation_id,
                "attempt_id": attempt_id,
                "game": "wa30",
                "target_level": 9,
                "frontier_sha256": frontier_hash,
                "input_tree_sha256": input_tree_sha256,
                "parent_source_tree_sha256":
                    parent_source_tree_sha256,
                "initial_workspace_tree_sha256":
                    initial_workspace_tree_sha256,
                "parent_checkpoint_sha256": parent_sha256,
                "wip_tree_sha256": None,
                "wip_solver_source_tree_sha256": None,
                "frontier_brief_sha256": frontier_brief_sha256,
                "bridge_policy_sha256": bridge_policy_sha256,
                "parent_action_count": 8,
                    "remaining_action_budget": 592,
                    "fresh_prefix_required": False,
                    "supervisory_handoff_sha256": None,
                    "supervisory_handoff_binding_receipt_sha256": None,
                },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    return R.AttemptSpec(
        schema=R.RUNNER_SCHEMA,
        campaign_id=campaign_id,
        generation_id=generation_id,
        attempt_id=attempt_id,
        game="wa30",
        target_level=9,
        authoritative_target=9,
        parent_checkpoint_path=str(parent),
        parent_checkpoint_sha256=parent_sha256,
        frontier_sha256=frontier_hash,
        generation_dir=str(generation),
        input_dir=str(generation / "input"),
        scratch_dir=str(generation / "scratch"),
        workspace_dir=str(generation / "scratch"),
        output_dir=str(generation / "output"),
        arena_socket_path=str(generation / "rpc" / "arena.sock"),
        arena_token_file_path=str(generation / "rpc" / "token"),
        bridge_dir=str(generation / "bridge"),
        bridge_socket_path=str(generation / "bridge" / "proposer.sock"),
        bridge_token_file_path=str(
            generation / "bridge" / "proposer-token"
        ),
        bridge_policy_receipt_path=str(
            generation / "host" / "bridge_policy_receipt.json"
        ),
        host_transcript_path=str(generation / "host" / "backend.jsonl"),
        app_server_transcript_path=str(
            generation / "host" / "app_server.jsonl"
        ),
        neutral_host_cwd_path=str(generation / "host" / "neutral"),
        app_server_state_dir=str(codex_home),
        app_server_control_dir=str(
            generation / "host" / "app_server_control"
        ),
        image_reference=IMAGE_REFERENCE,
        image_digest=MANIFEST_A,
        worker_command=R.EXPECTED_WORKER_COMMAND,
        resource_limits=R.ResourceLimitsProjection(
            cpus=1.5,
            memory_bytes=768 * 1024 * 1024,
            pids=96,
            tmpfs_bytes=64 * 1024 * 1024,
        ),
        proposer_transport=transport,
        input_tree_sha256=input_tree_sha256,
        parent_source_path=str(lane_parent_source),
        parent_source_tree_sha256=parent_source_tree_sha256,
        initial_workspace_tree_sha256=initial_workspace_tree_sha256,
        initial_app_server_state_tree_sha256=(
            initial_app_server_state_tree_sha256
        ),
        hard_safety_seconds=21_600,
        max_auth_refreshes=7,
        input_bundle_receipt_path=str(receipt),
        input_bundle_receipt_sha256=receipt_sha256,
        frontier_brief_path=str(frontier_brief),
        frontier_brief_sha256=frontier_brief_sha256,
        supervisory_handoff_path=None,
        supervisory_handoff_sha256=None,
        supervisory_handoff_binding_receipt_path=None,
        supervisory_handoff_binding_receipt_sha256=None,
        bridge_policy_path=str(bridge_policy),
        bridge_policy_sha256=bridge_policy_sha256,
        parent_action_count=8,
        remaining_action_budget=592,
        fresh_prefix_required=False,
        effort="max",
        soft_allocation_seconds=90 * 60,
        wip_mode="exclude",
        thread_mode="new",
        resume_thread_id=None,
        resume_thread_binding_sha256=None,
        wip=None,
        supervisory_handoff=None,
        cost_limit_remaining=None,
    )


def _short_runner_attempt_spec(request: pytest.FixtureRequest):
    return _runner_attempt_spec(
        _request_short_private_test_root(request)
    )


def _test_controller_canaries(prefix: str):
    return tuple(
        Taint.LiveCanary(
            category=category,
            location_name=f"{prefix}:{category}",
            value=hashlib.sha256(
                f"{prefix}:{category}:value".encode()
            ).hexdigest(),
        )
        for category in Taint.CONTROLLER_CANARY_CATEGORIES
    )


def _canary_backend(canaries):
    backend = B.ContiguousDockerAttemptBackend.__new__(
        B.ContiguousDockerAttemptBackend
    )
    backend._controller_state_canaries = canaries
    backend._attempt_controller_canaries = {}
    return backend


def _test_canary_anchor(spec, prefix: str = "runner-contract"):
    backend = _canary_backend(_test_controller_canaries(prefix))
    return backend._ensure_canary_anchor(spec)


def _explicit_unlimited_window(*, phase: str, sequence: int):
    response = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "team",
                "credits": {
                    "hasCredits": True,
                    "unlimited": True,
                    "balance": None,
                },
                "spendControlReached": False,
            },
        },
    }
    return T.normalize_provider_usage_window(
        response,
        phase=phase,
        observation_sequence=sequence,
        authenticated_response_sha256=hashlib.sha256(
            f"adapter-rate-response:{phase}:{sequence}".encode("ascii")
        ).hexdigest(),
        transcript_chain_sha256=hashlib.sha256(
            f"adapter-rate-chain:{phase}:{sequence}".encode("ascii")
        ).hexdigest(),
    )


class _AuthenticatedBridgeClientDouble:
    def __init__(self, spec, **_kwargs):
        from arc_agi3_arena_rpc import ArenaRpcClient

        transport = spec.proposer_transport
        self.challenge_nonce = "a" * 32
        self.handshake_request_sha256 = "b" * 64
        self.handshake_response_sha256 = "c" * 64
        self.handshake_result = {
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "frontier_sha256": spec.frontier_sha256,
            "policy_sha256": spec.bridge_policy_sha256,
            "operation_allowlist": list(
                transport.bridge_operation_allowlist
            ),
            "exec_allowlist": list(
                transport.bridge_exec_allowlist
            ),
            "provider_credential_names": [],
            "environment_names": [],
        }
        self.closed = False
        arena_token = Path(
            spec.arena_token_file_path
        ).read_text(encoding="ascii")
        # Model an orderly proposer-side Arena close.  The host supervisor
        # itself never impersonates the solver client.
        ArenaRpcClient(
            spec.arena_socket_path, arena_token
        ).close()

    def close(self):
        self.closed = True


class _AuthenticatedControllerDouble:
    CONTROLLER_ID = "1" * 64
    PROXY_ID = "2" * 64
    PID = 987_654

    def __init__(self, spec, *, credentials, **_kwargs):
        self.spec = spec
        self.credentials = credentials
        self.process = None
        self.preflight = None
        self.turn_start = None
        self.turn_final = None
        self.teardown = None
        self.transcript_chain_sha256 = None
        self.transcript_event_count = None

    @property
    def host_root(self):
        return Path(self.spec.host_transcript_path).parent

    def start_and_preflight(self):
        if self.preflight is not None:
            return self.preflight
        state_root = Path(self.spec.app_server_state_dir)
        initial_state = T.inventory_controller_state(state_root)
        launch_path = self.host_root / "controller_launch_receipt.json"
        guardian_path = self.host_root / "controller_guardian_start.json"
        readiness_path = (
            self.host_root / "controller_egress_readiness.json"
        )
        probe_path = (
            self.host_root / "controller_egress_live_probe.json"
        )
        launch_intent_sha256 = "3" * 64
        transport = self.spec.proposer_transport
        readiness_nonce = (
            B.DockerControllerContainerLauncher
            ._proxy_readiness_nonce(
                SimpleNamespace(
                    campaign_id=self.spec.campaign_id,
                    generation_id=self.spec.generation_id,
                    attempt_id=self.spec.attempt_id,
                    attempt_spec_sha256=(
                        Runner.proposer_attempt_binding_sha256(
                            self.spec
                        )
                    ),
                ),
                transport,
            )
        )
        B._write_private_json_new(
            readiness_path,
            {
                "schema": 1,
                "kind": "arc_agi3_controller_egress_readiness",
                "status": "READY",
                "campaign_id": self.spec.campaign_id,
                "generation_id": self.spec.generation_id,
                "attempt_id": self.spec.attempt_id,
                "policy": transport.controller_egress_policy,
                "policy_sha256":
                    transport.controller_egress_policy_sha256,
                "readiness_nonce": readiness_nonce,
                "guardian_pid": 1,
                "controller_uid": B.CONTROLLER_PROXY_UID,
                "listen": "127.0.0.1:19443",
                "iptables_rules_sha256": "a" * 64,
                "ip6tables_rules_sha256": "b" * 64,
                "allowed_sni":
                    list(B.CONTROLLER_PROXY_ALLOWED_SNI),
                "resolver_ipv4": ["127.0.0.11"],
                "default_deny_installed": True,
            },
        )
        B._write_private_json_new(
            probe_path,
            {
                "schema": 1,
                "kind":
                    "arc_agi3_controller_egress_live_probe",
                "policy": transport.controller_egress_policy,
                "policy_sha256":
                    transport.controller_egress_policy_sha256,
                "nonce": readiness_nonce,
                "uid": B.CONTROLLER_PROXY_UID,
                "checks": {
                    "allowed_openai_tls": True,
                    "denied_unallowlisted_sni": True,
                    "denied_loopback": True,
                    "denied_metadata": True,
                },
                "status": "PASS",
            },
        )
        B._write_private_json_new(
            launch_path,
            {
                "schema": 1,
                "kind": "arc_agi3_controller_launch",
                "campaign_id": self.spec.campaign_id,
                "generation_id": self.spec.generation_id,
                "attempt_id": self.spec.attempt_id,
                "attempt_spec_sha256":
                    Runner.proposer_attempt_binding_sha256(self.spec),
                "controller_container_id": self.CONTROLLER_ID,
                "controller_image_digest": (
                    self.spec.proposer_transport.controller_image_digest
                ),
                "egress_proxy_container_id": self.PROXY_ID,
                "egress_proxy_image_digest": (
                    self.spec.proposer_transport
                    .controller_egress_proxy_image_digest
                ),
                "egress_policy_sha256": (
                    transport.controller_egress_policy_sha256
                ),
                "egress_readiness_nonce": readiness_nonce,
                "egress_readiness_receipt_path":
                    str(readiness_path),
                "egress_readiness_receipt_sha256":
                    hashlib.sha256(
                        readiness_path.read_bytes()
                    ).hexdigest(),
                "egress_live_probe_receipt_path":
                    str(probe_path),
                "egress_live_probe_receipt_sha256":
                    hashlib.sha256(
                        probe_path.read_bytes()
                    ).hexdigest(),
                "egress_live_probe_before_controller_create":
                    True,
                "launch_intent_sha256": launch_intent_sha256,
                "authoritative_identity":
                    "controller_container_cgroup",
                "credentials_in_argv_or_env": False,
                "bridge_or_arena_mounts": 0,
            },
        )
        B._write_private_json_new(
            guardian_path,
            {
                "schema": 1,
                "kind": "arc_agi3_controller_guardian_start",
                "campaign_id": self.spec.campaign_id,
                "generation_id": self.spec.generation_id,
                "attempt_id": self.spec.attempt_id,
                "controller_container_id": self.CONTROLLER_ID,
                "egress_proxy_container_id": self.PROXY_ID,
                "native_workspace": _native_workspace_receipt(),
                "state_root_write_probe": {
                    "schema": 1,
                    "kind": "controller_state_root_write_probe",
                    "relative_path":
                        B.CONTROLLER_STATE_WRITE_PROBE_NAME,
                    "payload_sha256": hashlib.sha256(
                        B.CONTROLLER_STATE_WRITE_PROBE_PAYLOAD
                    ).hexdigest(),
                    "bytes": len(
                        B.CONTROLLER_STATE_WRITE_PROBE_PAYLOAD
                    ),
                    "runtime_uid": int(
                        transport.controller_user.split(":", 1)[0]
                    ),
                    "runtime_gid": int(
                        transport.controller_user.split(":", 1)[1]
                    ),
                    "fsync_file": True,
                    "fsync_directory": True,
                    "unlinked": True,
                    "status": "PASS",
                },
            },
        )
        database = state_root / T.CODEX_STATE_DATABASE_NAME
        database.write_bytes(T.SQLITE3_HEADER + b"\0" * 32)
        initialized_state = T.inventory_controller_state(state_root)
        database_sha256 = hashlib.sha256(
            database.read_bytes()
        ).hexdigest()
        launch_sha256 = hashlib.sha256(
            launch_path.read_bytes()
        ).hexdigest()
        guardian_sha256 = hashlib.sha256(
            guardian_path.read_bytes()
        ).hexdigest()
        self.preflight = T.PreflightEvidence(
            schema=1,
            pid=self.PID,
            process_group_id=self.PID,
            process_start_identity="4" * 64,
            codex_binary_sha256=transport.codex_binary_sha256,
            codex_binary_bytes=transport.codex_binary_bytes,
            initialize_params_sha256=T.INITIALIZE_PARAMS_SHA256,
            redacted_login_request_sha256=(
                self.credentials.redacted_request_sha256()
            ),
            dynamic_tool_specs_sha256=T.DYNAMIC_TOOL_SPECS_SHA256,
            base_instructions_sha256=T.BASE_INSTRUCTIONS_SHA256,
            developer_instructions_sha256=(
                T.DEVELOPER_INSTRUCTIONS_SHA256
            ),
            model=transport.model,
            model_provider=transport.model_provider,
            reasoning_effort=self.spec.effort,
            hard_safety_seconds=self.spec.hard_safety_seconds,
            max_auth_refreshes=self.spec.max_auth_refreshes,
            process_start_receipt_sha256=launch_sha256,
            process_identity_authority="controller_container_cgroup",
            controller_container_id=self.CONTROLLER_ID,
            controller_image_digest=transport.controller_image_digest,
            egress_proxy_container_id=self.PROXY_ID,
            egress_proxy_image_digest=(
                transport.controller_egress_proxy_image_digest
            ),
            egress_policy_sha256=(
                transport.controller_egress_policy_sha256
            ),
            controller_launch_intent_sha256=launch_intent_sha256,
            controller_launch_receipt_path=str(launch_path),
            controller_launch_receipt_sha256=launch_sha256,
            guardian_start_receipt_path=str(guardian_path),
            guardian_start_receipt_sha256=guardian_sha256,
            supply_chain_manifest_sha256="5" * 64,
            request_methods=tuple(
                transport.controller_preflight_request_allowlist
            ),
            notification_counts=tuple(
                sorted(T.PREFLIGHT_NOTIFICATION_CARDINALITY.items())
            ),
            response_sha256=tuple(
                (method, hashlib.sha256(method.encode("ascii")).hexdigest())
                for method in (
                    transport.controller_preflight_request_allowlist
                )
            ),
            provider_usage_window=_explicit_unlimited_window(
                phase="preflight", sequence=1
            ),
            auth_mode="chatgptAuthTokens",
            model_effort_supported=True,
            system_skills_disabled=True,
            hooks_empty=True,
            plugins_empty=True,
            apps_empty=True,
            experimental_features_disabled=True,
            mcp_servers_empty=True,
            stderr_empty=True,
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            stderr_bytes=0,
            path_alias_setup_status="PASS",
            state_root=str(state_root),
            initial_state_tree_sha256=initial_state.tree_sha256,
            initialized_state_tree_sha256=
                initialized_state.tree_sha256,
            initialized_state_inventory_sha256=
                initialized_state.inventory_sha256,
            initialized_state_file_count=
                len(initialized_state.files),
            initialized_state_total_bytes=
                initialized_state.total_bytes,
            state_database_path=T.CODEX_STATE_DATABASE_NAME,
            state_database_sha256=database_sha256,
            state_database_bytes=len(database.read_bytes()),
            state_database_header_sha256=hashlib.sha256(
                T.SQLITE3_HEADER
            ).hexdigest(),
            state_database_initialized=True,
            transcript_chain_sha256="6" * 64,
        )
        return self.preflight

    def start_turn(self, *, frontier_brief):
        assert frontier_brief["frontier_sha256"] == self.spec.frontier_sha256
        if self.turn_start is None:
            import test_arc_agi3_contiguous_runner as runner_test

            transcript_spec = dataclasses.replace(
                self.spec,
                app_server_state_dir=B.CONTROLLER_STATE_DESTINATION,
                neutral_host_cwd_path=B.CONTROLLER_NEUTRAL_DESTINATION,
            )
            thread_id = str(uuid.uuid4())
            turn_id = str(uuid.uuid4())
            (
                self.transcript_chain_sha256,
                self.transcript_event_count,
            ) = runner_test._write_app_transcript(
                Path(self.spec.app_server_transcript_path),
                transcript_spec,
                thread_id=thread_id,
                turn_id=turn_id,
            )
            scan_policy, _prompt = runner_test._app_scan_policy(
                transcript_spec
            )
            self.turn_start = T.TurnStartEvidence(
                schema=1,
                thread_id=thread_id,
                turn_id=turn_id,
                thread_mode=self.spec.thread_mode,
                thread_request_sha256="7" * 64,
                turn_request_sha256="8" * 64,
                prompt_sha256=scan_policy.prompt_sha256,
                transcript_chain_sha256=(
                    self.transcript_chain_sha256
                ),
            )
        return self.turn_start

    def run_turn(self):
        if self.turn_final is not None:
            return self.turn_final
        observations = ({
            "total": {
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "outputTokens": 1,
                "reasoningOutputTokens": 0,
                "totalTokens": 2,
            },
        },)
        pre = self.preflight.provider_usage_window
        post = _explicit_unlimited_window(
            phase="postflight", sequence=2
        )
        settlement = T.settle_provider_usage(
            pre,
            post,
            token_usage_observations=observations,
        )
        final_text = "authenticated controller test turn complete"
        self.turn_final = T.TurnFinalEvidence(
            schema=1,
            thread_id=self.turn_start.thread_id,
            turn_id=self.turn_start.turn_id,
            turn_status="completed",
            provider_outcome="completed",
            token_usage_observations=observations,
            pre_provider_usage_window=pre,
            post_provider_usage_window=post,
            provider_usage_settlement=settlement,
            final_model_text_sha256=hashlib.sha256(
                final_text.encode("utf-8")
            ).hexdigest(),
            final_model_text=final_text,
            tool_call_count=0,
            hard_safety_seconds=self.spec.hard_safety_seconds,
            max_auth_refreshes=self.spec.max_auth_refreshes,
            auth_refresh_count=0,
            redacted_auth_refresh_response_sha256=(),
            credential_sentinel_scan_passed=True,
            post_turn_event_count=0,
            stdout_bytes=0,
            stderr_bytes=0,
            pipes_drained_to_eof=True,
            transcript_chain_sha256=(
                self.transcript_chain_sha256
            ),
            transcript_event_count=self.transcript_event_count,
        )
        return self.turn_final

    def contain(self):
        if self.teardown is not None:
            return self.teardown
        if self.turn_final is None and self.turn_start is not None:
            self.run_turn()
        terminal_chain_sha256 = (
            self.preflight.transcript_chain_sha256
            if self.turn_final is None
            else self.turn_final.transcript_chain_sha256
        )
        absence_path = self.host_root / "controller_absence_receipt.json"
        B._write_private_json_new(
            absence_path,
            {
                "schema": 1,
                "kind": "arc_agi3_controller_absence",
                "campaign_id": self.spec.campaign_id,
                "generation_id": self.spec.generation_id,
                "attempt_id": self.spec.attempt_id,
                "attempt_spec_sha256":
                    Runner.proposer_attempt_binding_sha256(self.spec),
                "controller_container_id": self.CONTROLLER_ID,
                "egress_proxy_container_id": self.PROXY_ID,
                "controller_launch_receipt_sha256": (
                    self.preflight.controller_launch_receipt_sha256
                ),
                "guardian_start_receipt_sha256": (
                    self.preflight.guardian_start_receipt_sha256
                ),
                "authoritative_identity":
                    "controller_container_cgroup",
                "controller_inspect_absent": True,
                "controller_identity_query_empty": True,
                "controller_top_absent": True,
                "controller_no_descendants": True,
                "egress_proxy_inspect_absent": True,
                "egress_proxy_identity_query_empty": True,
                "egress_proxy_top_absent": True,
                "egress_proxy_no_descendants": True,
            },
        )
        absence_sha256 = hashlib.sha256(
            absence_path.read_bytes()
        ).hexdigest()
        state_sha256 = T.inventory_controller_state(
            Path(self.spec.app_server_state_dir)
        ).tree_sha256
        self.teardown = T.ControllerTeardownEvidence(
            schema=1,
            pid=self.PID,
            process_group_id=self.PID,
            exit_code=0,
            process_absent=True,
            process_group_absent=True,
            process_absent_receipt_sha256=absence_sha256,
            process_start_receipt_removed=True,
            ephemeral_tmp_purged=True,
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            stderr_bytes=0,
            state_tree_sha256=state_sha256,
            transcript_chain_sha256=(
                terminal_chain_sha256
            ),
            process_identity_authority="controller_container_cgroup",
            controller_container_id=self.CONTROLLER_ID,
            egress_proxy_container_id=self.PROXY_ID,
            controller_inspect_absent=True,
            controller_identity_query_empty=True,
            controller_top_absent=True,
            controller_no_descendants=True,
            egress_proxy_inspect_absent=True,
            egress_proxy_identity_query_empty=True,
            egress_proxy_top_absent=True,
            egress_proxy_no_descendants=True,
            controller_absence_receipt_sha256=absence_sha256,
        )
        return self.teardown

    def credential_sentinels_for_host_scan(self):
        return tuple(sorted(set(self.credentials.leak_sentinels)))


def _append_rejected_boundary_write_to_app_transcript(
    path: Path,
    *,
    thread_id: str,
    turn_id: str,
    arguments: dict[str, Any],
) -> tuple[str, int]:
    """Insert the exact app-server evidence left by a pre-write rejection."""

    retained = [
        (
            row["direction"],
            row["payload"],
        )
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    insertion = next(
        index
        for index, (direction, payload) in enumerate(retained)
        if direction == "server_notification"
        and isinstance(payload, dict)
        and payload.get("method") == "turn/completed"
    )
    call_id = "boundary-write-call"
    namespace = "contiguous_lane"
    tool = "workspace_write"
    retained[insertion:insertion] = [
        (
            "server_notification",
            {
                "method": "item/started",
                "params": {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "item": {
                        "arguments": dict(arguments),
                        "id": call_id,
                        "namespace": namespace,
                        "status": "inProgress",
                        "tool": tool,
                        "type": "dynamicToolCall",
                    },
                },
            },
        ),
        (
            "server_request",
            {
                "id": "boundary-write-request",
                "method": "item/tool/call",
                "params": {
                    "arguments": dict(arguments),
                    "callId": call_id,
                    "namespace": namespace,
                    "threadId": thread_id,
                    "tool": tool,
                    "turnId": turn_id,
                },
            },
        ),
        (
            "client_response",
            {
                "id": "boundary-write-request",
                "result": {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": json.dumps(
                                {"error": "AppServerTransportError"},
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                    "success": False,
                },
            },
        ),
        (
            "server_notification",
            {
                "method": "item/completed",
                "params": {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "item": {
                        "arguments": dict(arguments),
                        "id": call_id,
                        "namespace": namespace,
                        "status": "completed",
                        "success": False,
                        "tool": tool,
                        "type": "dynamicToolCall",
                    },
                },
            },
        ),
    ]
    previous: str | None = None
    rows: list[bytes] = []
    for sequence, (direction, payload) in enumerate(retained, 1):
        body = {
            "schema": 1,
            "sequence": sequence,
            "previous_digest": previous,
            "direction": direction,
            "payload": payload,
        }
        digest = hashlib.sha256(T.canonical_json(body)).hexdigest()
        rows.append(T.canonical_json({**body, "digest": digest}))
        previous = digest
    assert previous is not None
    path.write_bytes(b"\n".join(rows) + b"\n")
    return previous, len(rows)


class _BoundaryRejectedControllerDouble(_AuthenticatedControllerDouble):
    def __init__(self, spec, *, boundary_arguments, **kwargs):
        super().__init__(spec, **kwargs)
        self.boundary_arguments = dict(boundary_arguments)

    def start_turn(self, *, frontier_brief):
        evidence = super().start_turn(frontier_brief=frontier_brief)
        if getattr(self, "_boundary_write_recorded", False):
            return self.turn_start
        (
            self.transcript_chain_sha256,
            self.transcript_event_count,
        ) = _append_rejected_boundary_write_to_app_transcript(
            Path(self.spec.app_server_transcript_path),
            thread_id=evidence.thread_id,
            turn_id=evidence.turn_id,
            arguments=self.boundary_arguments,
        )
        self.turn_start = dataclasses.replace(
            evidence,
            transcript_chain_sha256=self.transcript_chain_sha256,
        )
        self._boundary_write_recorded = True
        return self.turn_start

    def run_turn(self):
        evidence = super().run_turn()
        observations = tuple(
            {
                **observation,
                "threadId": evidence.thread_id,
                "turnId": evidence.turn_id,
            }
            for observation in evidence.token_usage_observations
        )
        if (
            evidence.tool_call_count == 0
            or observations != evidence.token_usage_observations
        ):
            self.turn_final = dataclasses.replace(
                evidence,
                tool_call_count=1,
                token_usage_observations=observations,
                provider_usage_settlement=T.settle_provider_usage(
                    evidence.pre_provider_usage_window,
                    evidence.post_provider_usage_window,
                    token_usage_observations=observations,
                ),
            )
        return self.turn_final


def _authenticated_adapter_test_kwargs(
    low_backend: B.DockerContainerBackend,
    spec,
    *,
    canary_prefix: str = "runner-contract",
) -> dict[str, Any]:
    """Supply the current authenticated controller/probe constructor edge."""

    probe_executor = B.DockerWorkspaceProbeExecutor.__new__(
        B.DockerWorkspaceProbeExecutor
    )
    probe_executor._backend = low_backend
    probe_executor._query_attempt_probe_ids = lambda _labels: ()
    credentials = T.ExternalChatGptCredentials(
        access_token="adapter-test-access-token",
        account_id="adapter-test-account",
        plan_type=None,
        leak_sentinels=("adapter-test-access-token",),
        source_path="/dev/null",
    )
    arena_lifecycle = getattr(
        low_backend, "_test_arena_volume_lifecycle", None
    )
    if arena_lifecycle is None:
        arena_runner = _ArenaVolumeDockerDouble(
            B.AttemptIdentity(
                campaign_id=spec.campaign_id,
                generation_id=spec.generation_id,
                attempt_id=spec.attempt_id,
                game=spec.game,
                target_level=spec.target_level,
            )
        )
        arena_lifecycle = B.DockerArenaVolumeLifecycle(
            B.DockerContainerBackend(arena_runner)
        )
        arena_lifecycle._start_attachment = (
            lambda *, container_id, arena_socket:
            _RelayAttachmentDouble()
        )
        low_backend._test_arena_volume_lifecycle = arena_lifecycle
    return {
        "credentials": credentials,
        "probe_executor": probe_executor,
        # These lifecycle tests use receipt-producing controller/bridge
        # doubles. Selecting a non-production controller keeps the static test
        # canaries explicit while preserving every authenticated host edge.
        "controller_factory": lambda **kwargs:
            _AuthenticatedControllerDouble(
                kwargs["probe_spec"], **kwargs
            ),
        "bridge_client_factory": lambda **kwargs:
            _AuthenticatedBridgeClientDouble(spec, **kwargs),
        "controller_state_canaries":
            _test_controller_canaries(canary_prefix),
        "arena_volume_lifecycle": arena_lifecycle,
    }


def _prepare_only_adapter(spec):
    from arc_agi3_arena_rpc import ArenaHostSession

    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)
    adapter = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args, **_kwargs: None,
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=lambda game, *, binding, parent_path, token:
        ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        ),
    )
    return adapter, low_backend


def test_canary_escrow_survives_crash_between_launch_and_teardown(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    original = _test_controller_canaries("original")
    first = _canary_backend(original)
    assert first._ensure_canary_escrow(spec) == Taint.validate_live_canaries(
        original
    )

    escrow = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_escrow"
        / f"{spec.generation_id}.json"
    )
    assert escrow.stat().st_mode & 0o777 == 0o400
    assert Path(spec.generation_dir) not in escrow.parents

    # Simulate a new host supervisor with a newly generated in-memory set.
    restarted = _canary_backend(_test_controller_canaries("replacement"))
    recovered = restarted._ensure_canary_escrow(spec)
    assert recovered == Taint.validate_live_canaries(original)
    assert recovered != Taint.validate_live_canaries(
        _test_controller_canaries("replacement")
    )


def test_canary_anchor_rejects_missing_and_substituted_escrow(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    backend = _canary_backend(_test_controller_canaries("anchor"))
    anchor = backend._ensure_canary_anchor(spec)
    anchor.validate()
    escrow = Path(anchor.escrow_path)
    displaced = escrow.with_name("displaced.json")
    escrow.rename(displaced)
    with pytest.raises(
        B.ContainerContractError,
        match="missing path component",
    ):
        anchor.validate()
    displaced.rename(escrow)
    os.chmod(escrow, 0o600)
    with pytest.raises(
        B.ContainerContractError,
        match="changed",
    ):
        anchor.validate()


def test_canary_anchor_rejects_path_swap_during_descriptor_read(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _short_runner_attempt_spec(request)
    backend = _canary_backend(_test_controller_canaries("path-swap"))
    backend._ensure_canary_escrow(spec)
    escrow = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_escrow"
        / f"{spec.generation_id}.json"
    )
    replacement = escrow.with_name("replacement.json")
    replacement.write_text(
        escrow.read_text(encoding="utf-8").replace(
            "path-swap", "substitute"
        ),
        encoding="utf-8",
    )
    os.chmod(replacement, 0o400)
    real_read = B.os.read
    swapped = False

    def swap_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        payload = real_read(descriptor, size)
        if payload and not swapped:
            os.replace(replacement, escrow)
            swapped = True
        return payload

    monkeypatch.setattr(B.os, "read", swap_after_first_read)
    with pytest.raises(
        B.ContainerContractError,
        match="changed",
    ):
        backend._ensure_canary_anchor(spec)
    assert swapped is True


def test_direct_proposer_backend_rejects_missing_canary_anchor_before_create(
    request: pytest.FixtureRequest,
):
    short_root = _request_private_test_root(
        request, prefix=".a3d_"
    )
    spec = _runner_attempt_spec(short_root)
    backend = _canary_backend(_test_controller_canaries("direct"))
    anchor = backend._ensure_canary_anchor(spec)
    low = B.AttemptSpec(
        identity=B.AttemptIdentity(
            campaign_id=spec.campaign_id,
            generation_id=spec.generation_id,
            attempt_id=spec.attempt_id,
            game=spec.game,
            target_level=spec.target_level,
        ),
        image_reference=spec.image_reference,
        parent_input=Path(spec.input_dir),
        export_root=Path(spec.output_dir),
        arena_socket=Path(spec.arena_socket_path),
        arena_token_file=Path(spec.arena_token_file_path),
        command=B.expected_proposer_worker_command(),
        resource_limits=B.ResourceLimits(
            cpus=spec.resource_limits.cpus,
            memory_bytes=spec.resource_limits.memory_bytes,
            pids=spec.resource_limits.pids,
            tmpfs_bytes=spec.resource_limits.tmpfs_bytes,
        ),
        soft_allocation_seconds=spec.soft_allocation_seconds,
        role="proposer",
        workspace_root=Path(spec.workspace_dir),
        bridge_root=Path(spec.bridge_dir),
        bridge_socket=Path(spec.bridge_socket_path),
        bridge_token_file=Path(spec.bridge_token_file_path),
        containment_canary_anchor=anchor,
    )
    missing = dataclasses.replace(
        low, containment_canary_anchor=None
    )
    docker_runner = FakeDockerRunner(missing)
    with pytest.raises(
        B.ContainerContractError,
        match="lacks an exact prelaunch canary anchor",
    ):
        B.DockerContainerBackend(
            docker_runner
        ).build_launch_attestation(missing)
    assert docker_runner.commands == []


def test_terminal_canary_reveal_is_post_containment_and_idempotent(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R

    spec = _short_runner_attempt_spec(request)
    canaries = _test_controller_canaries("terminal")
    backend = _canary_backend(canaries)
    anchor = backend._ensure_canary_anchor(spec)
    prepared = R.BackendPreparation(
        preparation_id="canary-test",
        launch_attestation_path=str(
            Path(spec.host_transcript_path).parent
            / "launch_attestation.json"
        ),
        launch_attestation_sha256="a" * 64,
        observed_image_digest=MANIFEST_A,
        image_observation_sha256="b" * 64,
        container_observation_sha256="c" * 64,
        bridge_policy_receipt_path=spec.bridge_policy_receipt_path,
        bridge_policy_receipt_sha256="d" * 64,
        arena_session_binding_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "arena_session_binding_receipt.json"
        ),
        arena_session_binding_receipt_sha256="9" * 64,
        compatibility_closure_path=str(
            Path(spec.host_transcript_path).parent
            / B.COMPATIBILITY_CLOSURE_DIRECTORY
        ),
        compatibility_closure_receipt_sha256="b" * 64,
        compatibility_turn_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / B.COMPATIBILITY_TURN_RECEIPT_NAME
        ),
        compatibility_turn_receipt_sha256="c" * 64,
        arena_transport=B.ARENA_VOLUME_TRANSPORT,
        arena_volume_name=B.arena_volume_name(
            B.AttemptIdentity(
                campaign_id=spec.campaign_id,
                generation_id=spec.generation_id,
                attempt_id=spec.attempt_id,
                game=spec.game,
                target_level=spec.target_level,
            )
        ),
        arena_volume_observation_sha256="4" * 64,
        arena_relay_container_id=RELAY_CONTAINER_ID,
        arena_relay_image_digest=RELAY_MANIFEST,
        arena_relay_image_observation_sha256="5" * 64,
        arena_relay_container_observation_sha256="6" * 64,
        arena_relay_readiness_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "arena_volume_readiness.json"
        ),
        arena_relay_readiness_receipt_sha256="7" * 64,
        arena_relay_attach_argv_sha256="8" * 64,
        arena_relay_socket_identity_sha256="9" * 64,
        arena_relay_preparation_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "arena_volume_preparation.json"
        ),
        arena_relay_preparation_receipt_sha256="a" * 64,
        probe_isolation_mode=(
            R.Contract.VERIFIED_ISOLATED_CLONE_MODE
        ),
        probe_isolation_evidence_sha256="8" * 64,
        neutral_cwd_attestation_path=str(
            Path(spec.neutral_host_cwd_path).parent
            / "neutral_cwd_attestation.json"
        ),
        neutral_cwd_attestation_sha256="e" * 64,
        app_server_config_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "app_server_config_receipt.json"
        ),
        app_server_config_receipt_sha256="f" * 64,
        codex_binary_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "codex_binary_receipt.json"
        ),
        codex_binary_receipt_sha256="1" * 64,
        protocol_schema_receipt_path=str(
            Path(spec.host_transcript_path).parent
            / "app_server_protocol_schema_receipt.json"
        ),
        protocol_schema_receipt_sha256="2" * 64,
        controller_image_digest=MANIFEST_A,
        controller_egress_proxy_image_digest=MANIFEST_B,
        controller_egress_policy_sha256="3" * 64,
        controller_canary_escrow_path=anchor.escrow_path,
        controller_canary_escrow_sha256=anchor.escrow_sha256,
        controller_canary_escrow_identity_sha256=(
            anchor.escrow_identity_sha256
        ),
        controller_canary_commitments_json=anchor.commitments_json,
        controller_canary_commitments_sha256=(
            anchor.commitments_sha256
        ),
        controller_canary_placement_descriptors_json=(
            anchor.placement_descriptors_json
        ),
        controller_canary_placement_descriptors_sha256=(
            anchor.placement_descriptors_sha256
        ),
        controller_supply_chain_unobserved_until_launch=True,
    )
    commitments = backend._canary_commitment_documents(
        Taint.validate_live_canaries(canaries)
    )
    host = Path(spec.host_transcript_path).parent
    state_scan_path = host / "controller_state_scan_receipt.json"
    retained_scan_path = (
        Path(spec.generation_dir)
        / "retained_canary_scan_receipt.json"
    )
    B._write_private_json_new(
        state_scan_path,
        {
            "controller_state_scan": {
                "canary_commitments": commitments,
                "canary_occurrences": 0,
                "status": "CLEAN",
            }
        },
    )
    B._write_private_json_new(
        retained_scan_path,
        {
            "retained_canary_scan": {
                "canary_commitments": commitments,
                "canary_occurrences": 0,
                "status": "CLEAN",
            }
        },
    )
    collection = SimpleNamespace(
        controller_state_scan_receipt_path=str(state_scan_path),
        controller_state_scan_receipt_sha256=hashlib.sha256(
            state_scan_path.read_bytes()
        ).hexdigest(),
        retained_canary_scan_receipt_path=str(retained_scan_path),
        retained_canary_scan_receipt_sha256=hashlib.sha256(
            retained_scan_path.read_bytes()
        ).hexdigest(),
    )
    controller_id = "1" * 64
    proxy_id = "2" * 64
    proposer_id = "3" * 64
    B._write_private_json_new(
        host / "controller_absence_receipt.json",
        {
            "schema": 1,
            "kind": "arc_agi3_controller_absence",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "attempt_spec_sha256":
                R.proposer_attempt_binding_sha256(spec),
            "controller_container_id": controller_id,
            "egress_proxy_container_id": proxy_id,
            "controller_launch_receipt_sha256": "4" * 64,
            "guardian_start_receipt_sha256": "5" * 64,
            "guardian_exit_receipt_sha256": "6" * 64,
            "controller_inspect_absent": True,
            "controller_identity_query_empty": True,
            "controller_top_absent": True,
            "controller_no_descendants": True,
            "egress_proxy_inspect_absent": True,
            "egress_proxy_identity_query_empty": True,
            "egress_proxy_top_absent": True,
            "egress_proxy_no_descendants": True,
            "authoritative_identity":
                "controller_container_cgroup",
        },
    )
    B._write_private_json_new(
        host / "probe_reconciliation_teardown.json",
        {
            "schema": 1,
            "kind": "arc_agi3_contiguous_probe_reconciliation",
            "stage": "teardown",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "removed_container_ids": [],
            "identity_query_empty": True,
            "container_inspect_absent": True,
            "process_groups_absent": True,
        },
    )

    class ExactAbsenceBackend:
        _docker = "docker"

        def __init__(self):
            self.absent = False

        def _expect_absent(self, *_args, **_kwargs):
            return self.absent

        @staticmethod
        def _identity_query(_identity):
            return B.CommandResult((), 0, "", "")

        @staticmethod
        def _required(argv, **_kwargs):
            return B.CommandResult(tuple(argv), 0, "", "")

    class ExactProbeExecutor:
        @staticmethod
        def _query_attempt_probe_ids(_labels):
            return ()

    absence_backend = ExactAbsenceBackend()
    backend._backend = absence_backend
    backend._probe_executor = ExactProbeExecutor()
    reveal_path = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_reveals"
        / f"{spec.generation_id}.json"
    )
    with pytest.raises(
        B.ContainerContractError,
        match="proposer identity is not absent",
    ):
        backend._ensure_terminal_canary_reveal(
            spec=spec,
            prepared=prepared,
            controller_state_scan_receipt_path=(
                collection.controller_state_scan_receipt_path
            ),
            controller_state_scan_receipt_sha256=(
                collection.controller_state_scan_receipt_sha256
            ),
            retained_canary_scan_receipt_path=(
                collection.retained_canary_scan_receipt_path
            ),
            retained_canary_scan_receipt_sha256=(
                collection.retained_canary_scan_receipt_sha256
            ),
            container_id=proposer_id,
            controller_container_id=controller_id,
            egress_proxy_container_id=proxy_id,
            container_proof_sha256="7" * 64,
        )
    assert not reveal_path.exists()
    absence_backend.absent = True
    first = backend._ensure_terminal_canary_reveal(
        spec=spec,
        prepared=prepared,
        controller_state_scan_receipt_path=(
            collection.controller_state_scan_receipt_path
        ),
        controller_state_scan_receipt_sha256=(
            collection.controller_state_scan_receipt_sha256
        ),
        retained_canary_scan_receipt_path=(
            collection.retained_canary_scan_receipt_path
        ),
        retained_canary_scan_receipt_sha256=(
            collection.retained_canary_scan_receipt_sha256
        ),
        container_id=proposer_id,
        controller_container_id=controller_id,
        egress_proxy_container_id=proxy_id,
        container_proof_sha256="7" * 64,
    )
    assert Path(first[0]).stat().st_mode & 0o777 == 0o400

    restarted = _canary_backend(_test_controller_canaries("replacement"))
    restarted._backend = absence_backend
    restarted._probe_executor = ExactProbeExecutor()
    second = restarted._ensure_terminal_canary_reveal(
        spec=spec,
        prepared=prepared,
        controller_state_scan_receipt_path=(
            collection.controller_state_scan_receipt_path
        ),
        controller_state_scan_receipt_sha256=(
            collection.controller_state_scan_receipt_sha256
        ),
        retained_canary_scan_receipt_path=(
            collection.retained_canary_scan_receipt_path
        ),
        retained_canary_scan_receipt_sha256=(
            collection.retained_canary_scan_receipt_sha256
        ),
        container_id=proposer_id,
        controller_container_id=controller_id,
        egress_proxy_container_id=proxy_id,
        container_proof_sha256="7" * 64,
    )
    assert second == first
    reveal = json.loads(Path(first[0]).read_text(encoding="utf-8"))
    assert reveal["canary_commitments"] == commitments
    assert Taint.validate_live_canary_reveal(
        reveal["reveal"],
        expected_commitments=tuple(
            (
                row["category"],
                row["location_name"],
                row["provenance"],
                row["commitment_sha256"],
            )
            for row in commitments
        ),
    ) == Taint.validate_live_canaries(canaries)


def test_runner_backend_security_field_sets_are_exact(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R

    fields = {field.name for field in dataclasses.fields(R.AttemptSpec)}
    assert fields == B.RUNNER_ATTEMPT_SECURITY_FIELDS
    spec = _short_runner_attempt_spec(request)
    low = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    assert low.identity.attempt_id == spec.attempt_id
    assert low.identity.generation_id == spec.generation_id
    assert low.image_reference == spec.image_reference
    assert low.command == B.expected_proposer_worker_command()
    assert low.role == "proposer"
    assert low.workspace_root == Path(spec.workspace_dir)
    assert low.bridge_root == Path(spec.bridge_dir)
    assert low.soft_allocation_seconds == spec.soft_allocation_seconds


def test_proposer_create_mounts_and_attests_exact_lane_channels(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    low = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(low.arena_socket))
    request.addfinalizer(listener.close)
    low.arena_socket.chmod(0o600)
    low.arena_token_file.write_text("a" * 64, encoding="ascii")
    low.arena_token_file.chmod(0o600)
    low.bridge_token_file.write_text("b" * 64, encoding="ascii")
    low.bridge_token_file.chmod(0o600)

    backend, runner, attestation = _build(low)
    create = next(
        command for command in runner.commands if command[2] == "create"
    )
    mounts = [
        create[index + 1]
        for index, value in enumerate(create)
        if value == "--mount"
    ]
    assert len(mounts) == 7
    assert any(
        f"src={low.workspace_root},dst={B.WORKSPACE_DESTINATION},"
        "bind-propagation=rprivate"
        in mount
        for mount in mounts
    )
    assert any(
        f"src={low.bridge_root},dst={B.BRIDGE_ROOT_DESTINATION},"
        "bind-propagation=rprivate"
        in mount
        for mount in mounts
    )
    assert any(
        f"src={low.bridge_token_file},dst={B.BRIDGE_TOKEN_DESTINATION},"
        "readonly" in mount
        for mount in mounts
    )
    assert any(
        mount
        == (
            f"type=volume,src={B.arena_volume_name(low.identity)},"
            f"dst={B.PROPOSER_RPC_ROOT_DESTINATION},"
            "readonly,volume-nocopy"
        )
        for mount in mounts
    )
    assert not any(
        f"src={low.arena_socket}" in mount for mount in mounts
    )
    assert create[create.index("--workdir") + 1] == (
        B.WORKSPACE_DESTINATION
    )
    assert low.command == (
        "-I",
        "-m",
        "arc_agi3_proposer_worker",
        "--bridge-socket=/run/arc-agi3/proposer.sock",
        "--bridge-token-file=/run/arc-agi3/proposer-token",
        "--bridge-policy=/arc/input/bridge_policy.json",
        "--arena-socket=/arena/arena.sock",
        "--arena-token-file=/run/arc-agi3/token",
        "--workspace=/arc/workspace",
        "--export=/arc/export",
    )
    passed_environment = {
        create[index + 1].split("=", 1)[0]
        for index, value in enumerate(create)
        if value == "--env"
    }
    assert passed_environment == B.ATTEMPT_ENV_NAMES
    assert not any(
        token in name
        for name in passed_environment
        for token in ("AUTH", "CHATGPT", "CODEX", "OPENAI", "TOKEN")
    )
    assert attestation.role == "proposer"
    assert attestation.workspace_root is not None
    assert attestation.bridge_root is not None
    assert attestation.bridge_token_file is not None
    assert {fact.name for fact in attestation.evidence} == set(
        B.PROPOSER_ISOLATION_EVIDENCE_NAMES
    )

    def remove_workspace(record: dict[str, Any]) -> None:
        record["Mounts"] = [
            mount
            for mount in record["Mounts"]
            if mount["Destination"] != B.WORKSPACE_DESTINATION
        ]

    tampered = FakeDockerRunner(low, container_mutator=remove_workspace)
    with pytest.raises(B.ContainerContractError, match="exact role"):
        B.DockerContainerBackend(tampered).build_launch_attestation(low)


def test_runner_backend_contract_rejects_dropped_or_drifted_fields(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    incomplete = vars(spec).copy()
    incomplete.pop("frontier_sha256")
    with pytest.raises(B.ContainerContractError, match="drops.*frontier"):
        B.validate_runner_attempt_contract(
            SimpleNamespace(**incomplete),
            containment_canary_anchor=_test_canary_anchor(spec),
        )
    extended = vars(spec).copy()
    extended["caller_claims_isolated"] = True
    with pytest.raises(B.ContainerContractError, match="unreviewed fields"):
        B.validate_runner_attempt_contract(
            SimpleNamespace(**extended),
            containment_canary_anchor=_test_canary_anchor(spec),
        )

    with pytest.raises(B.ContainerContractError, match="frontier hash"):
        B.validate_runner_attempt_contract(
            dataclasses.replace(spec, frontier_sha256="f" * 64),
            containment_canary_anchor=_test_canary_anchor(spec),
        )

    Path(spec.input_dir, "solve.py").write_text(
        "def solve(env):\n    raise RuntimeError\n",
        encoding="utf-8",
    )
    with pytest.raises(B.ContainerContractError, match="input bundle hash"):
        B.validate_runner_attempt_contract(
            spec,
            containment_canary_anchor=_test_canary_anchor(spec),
        )


def test_runner_backend_contract_rejects_symlinked_generation_alias(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    alias_parent = (
        Path(spec.generation_dir).parent.parent / "aliased-attempts"
    ).resolve()
    alias_parent.symlink_to(Path(spec.generation_dir).parent)
    aliased_generation = alias_parent / spec.generation_id
    aliased = dataclasses.replace(
        spec,
            generation_dir=str(aliased_generation),
            input_dir=str(aliased_generation / "input"),
            scratch_dir=str(aliased_generation / "scratch"),
            workspace_dir=str(aliased_generation / "scratch"),
            output_dir=str(aliased_generation / "output"),
            arena_socket_path=str(aliased_generation / "rpc" / "arena.sock"),
            arena_token_file_path=str(aliased_generation / "rpc" / "token"),
            bridge_dir=str(aliased_generation / "bridge"),
            bridge_socket_path=str(
                aliased_generation / "bridge" / "proposer.sock"
            ),
            bridge_token_file_path=str(
                aliased_generation / "bridge" / "proposer-token"
            ),
            bridge_policy_receipt_path=str(
                aliased_generation / "host"
                / "bridge_policy_receipt.json"
            ),
            host_transcript_path=str(
                aliased_generation / "host" / "backend.jsonl"
            ),
            app_server_transcript_path=str(
                aliased_generation / "host" / "app_server.jsonl"
            ),
            neutral_host_cwd_path=str(
                aliased_generation / "host" / "neutral"
            ),
            app_server_state_dir=str(
                aliased_generation / "state" / "codex_home"
            ),
            app_server_control_dir=str(
                aliased_generation / "host" / "app_server_control"
            ),
            input_bundle_receipt_path=str(
                aliased_generation / "input_bundle_receipt.json"
            ),
            frontier_brief_path=str(
                aliased_generation / "input" / "frontier_brief.json"
            ),
            bridge_policy_path=str(
                aliased_generation / "input" / "bridge_policy.json"
            ),
    )
    with pytest.raises(B.ContainerContractError, match="symlinked"):
        B.validate_runner_attempt_contract(
            aliased,
            containment_canary_anchor=_test_canary_anchor(spec),
        )


def test_runner_backend_rejects_nonportable_rpc_socket_path(tmp_path: Path):
    spec = _runner_attempt_spec(tmp_path)
    assert len(os.fsencode(spec.arena_socket_path)) > (
        B.MAX_PORTABLE_UNIX_SOCKET_PATH_BYTES
    )
    with pytest.raises(B.ContainerContractError, match="portable Unix-domain"):
        B.validate_runner_attempt_contract(
            spec,
            containment_canary_anchor=_test_canary_anchor(spec),
        )


class _TinyArena:
    def __init__(self):
        self.actions = (1,)
        self.levels_completed = 0
        self.path: list[int] = []

    def terminal(self):
        return False

    def frame(self):
        return [[0]]

    def reset(self):
        self.path.clear()

    def step(self, action, x=None, y=None):
        self.path.append(action if x is None else (action, x, y))
        self.levels_completed += 1

    def clone(self):
        clone = _TinyArena()
        clone.levels_completed = self.levels_completed
        clone.path = list(self.path)
        return clone


def _tiny_arena_session_factory(
    game: str,
    *,
    binding: Any,
    parent_path: Sequence[Any],
    token: str,
):
    from arc_agi3_arena_rpc import ArenaHostSession

    return ArenaHostSession(
        game,
        binding=binding,
        parent_path=parent_path,
        token=token,
        arena_factory=lambda _game: _TinyArena(),
    )


def test_backend_binds_exact_parent_path_into_default_arena_contract(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    checkpoint, parent_path = B._load_exact_parent_checkpoint(spec)
    binding = B._arena_session_binding(
        spec, parent_level=checkpoint.reached
    )
    session = _tiny_arena_session_factory(
        spec.game,
        binding=binding,
        parent_path=parent_path,
        token="a" * 64,
    )
    projection = B._validate_arena_binding_event(
        session, spec=spec, parent_path=parent_path
    )
    assert projection["exploration_mode"] == "continue_parent"
    assert projection["parent_level"] == 8
    assert projection["target_level"] == 9
    assert projection["parent_replay_steps"] == 8
    assert projection["parent_checkpoint_sha256"] == (
        spec.parent_checkpoint_sha256
    )
    assert projection["binding_sha256"] == session.binding_sha256
    assert len(projection["seed_snapshot_sha256"]) == 64
    assert len(projection["exploration_seed_snapshot_sha256"]) == 64


def test_backend_arena_checkpoint_read_fails_closed_on_byte_drift(
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    parent = Path(spec.parent_checkpoint_path)
    parent.write_bytes(parent.read_bytes() + b" ")
    with pytest.raises(
        B.ContainerContractError,
        match="admitted digest",
    ):
        B._load_exact_parent_checkpoint(spec)


def test_host_shutdown_cannot_manufacture_clean_arena_close(
    request: pytest.FixtureRequest,
):
    import arc_agi3_arena_rpc as A

    spec = _short_runner_attempt_spec(request)
    checkpoint, parent_path = B._load_exact_parent_checkpoint(spec)
    session = _tiny_arena_session_factory(
        spec.game,
        binding=B._arena_session_binding(
            spec, parent_level=checkpoint.reached
        ),
        parent_path=parent_path,
        token="b" * 64,
    )
    server = A.ArenaRpcServer(
        session,
        Path(spec.arena_socket_path),
        Path(spec.host_transcript_path),
    )
    thread = server.start_thread()
    clean = B._finish_arena_server(
        session=session,
        server=server,
        thread=thread,
        socket_path=Path(spec.arena_socket_path),
        client_factory=lambda *_args: pytest.fail(
            "host teardown must not instantiate an Arena client"
        ),
    )
    assert clean is False
    with pytest.raises(
        A.ArenaRpcContractError,
        match="authenticated clean session close",
    ):
        session.host_result()
    events = [
        json.loads(line)
        for line in Path(spec.host_transcript_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1] == {
        "schema": 1,
        "kind": "arena_forced_shutdown",
        "clean_close_eligible": False,
    }


def test_arena_host_result_is_strictly_revalidated_before_collection(
    request: pytest.FixtureRequest,
):
    import arc_agi3_arena_rpc as A

    spec = _short_runner_attempt_spec(request)
    checkpoint, parent_path = B._load_exact_parent_checkpoint(spec)
    token = "c" * 64
    session = _tiny_arena_session_factory(
        spec.game,
        binding=B._arena_session_binding(
            spec, parent_level=checkpoint.reached
        ),
        parent_path=parent_path,
        token=token,
    )
    event = B._validate_arena_binding_event(
        session, spec=spec, parent_path=parent_path
    )
    B._write_private_token(Path(spec.arena_token_file_path), token)
    B._ensure_bound_receipt(
        Path(spec.host_transcript_path).parent
        / "arena_session_binding_receipt.json",
        spec=spec,
        kind="contiguous_arena_session_binding",
        fields={"binding_event": event},
    )
    valid = A.ArenaHostResult(
        binding_sha256=event["binding_sha256"],
        game=spec.game,
        exploration_mode="continue_parent",
        parent_level=8,
        levels_completed=8,
        parent_path=tuple(parent_path),
        path=tuple(parent_path),
        parent_replay_steps=8,
        exploration_steps=0,
        resets=0,
        total_steps=8,
        parent_terminal=False,
        parent_snapshot_sha256=event["seed_snapshot_sha256"],
    )
    B._validate_arena_host_result(valid, spec=spec)
    with pytest.raises(
        B.ContainerContractError,
        match="identity or accounting",
    ):
        B._validate_arena_host_result(
            dataclasses.replace(valid, total_steps=9),
            spec=spec,
        )
    with pytest.raises(
        B.ContainerContractError,
        match="trusted schema",
    ):
        B._validate_arena_host_result(
            SimpleNamespace(**dataclasses.asdict(valid)),
            spec=spec,
        )


def test_host_terminal_parent_issues_idempotent_authenticated_blocker(
    request: pytest.FixtureRequest,
):
    import arc_agi3_arena_rpc as A
    import arc_agi3_contiguous_runner as R

    class TerminalParentArena(_TinyArena):
        def terminal(self):
            return self.levels_completed >= 8

        def clone(self):
            clone = TerminalParentArena()
            clone.levels_completed = self.levels_completed
            clone.path = list(self.path)
            return clone

    spec = _short_runner_attempt_spec(request)
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(
            spec, "blocker-receipt"
        ),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)
    observed_host_results: list[A.ArenaHostResult] = []

    def collect_result(
        runner_spec, terminal, arena_result, worker_outcome, output_root
    ):
        assert runner_spec == spec
        assert terminal.status == "exited"
        assert isinstance(arena_result, A.ArenaHostResult)
        assert worker_outcome["authoritative"] is False
        assert output_root == Path(spec.output_dir)
        observed_host_results.append(arena_result)
        return R.AttemptResult(
            kind="clean_no_progress",
            reason="collector did not declare a blocker",
        )

    adapter = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=collect_result,
        **_authenticated_adapter_test_kwargs(
            low_backend, spec, canary_prefix="blocker-receipt"
        ),
        arena_session_factory=lambda game, *, binding, parent_path, token:
        A.ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: TerminalParentArena(),
        ),
    )
    prepared = adapter.prepare(spec)
    socket_path = Path(spec.arena_socket_path)
    assert socket_path.exists()
    launched = adapter.launch(spec, prepared)
    runtime = adapter._attempts[spec.attempt_id]
    runtime.arena_server.wait(timeout=5)
    assert not socket_path.exists()

    machine_result = runtime.arena_session.host_result()
    assert machine_result.parent_terminal is True
    assert machine_result.levels_completed == spec.target_level - 1
    B._validate_arena_host_result(machine_result, spec=spec)
    assert B._host_blocker_code(
        machine_result, spec=spec
    ) == "arena_parent_terminal_before_target"

    Path(spec.output_dir, "worker_outcome.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_proposer_worker",
                "attempt_id": spec.attempt_id,
                "authoritative": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    docker_runner.running = False
    terminal = adapter.poll(
        spec=spec,
        prepared=prepared,
        launched=launched,
        timeout_seconds=1,
    )
    assert terminal.status == "exited"
    collection = adapter.collect(
        spec=spec,
        prepared=prepared,
        launched=launched,
        terminal=terminal,
    )
    assert observed_host_results == [machine_result]
    assert collection.result.kind == "blocker"
    assert collection.result.blocker is not None
    assert collection.result.reason == (
        "host_blocker:arena_parent_terminal_before_target"
    )
    assert adapter.collect(
        spec=spec,
        prepared=prepared,
        launched=launched,
        terminal=terminal,
    ) == collection

    first = collection.result.blocker
    canaries = _test_controller_canaries("blocker-receipt")
    second = B._ensure_host_blocker_receipt(
        spec=spec,
        arena_host_result=machine_result,
        code="arena_parent_terminal_before_target",
        canaries=canaries,
    )
    assert first == second
    receipt_path = Path(first.receipt_path)
    receipt_raw = receipt_path.read_bytes()
    assert hashlib.sha256(receipt_raw).hexdigest() == first.receipt_sha256
    assert b"host_authentication_sha256" in receipt_raw
    assert all(
        canary.value.encode("ascii") not in receipt_raw
        for canary in canaries
    )
    receipt = json.loads(receipt_raw)
    unsigned = dict(receipt)
    authentication = unsigned.pop("host_authentication_sha256")
    assert authentication == R.host_blocker_authentication_sha256(
        unsigned, canaries
    )
    assert receipt["arena_host_result"] == json.loads(
        json.dumps(dataclasses.asdict(machine_result))
    )
    assert receipt["arena_host_result_sha256"] == hashlib.sha256(
        R._canonical_json(receipt["arena_host_result"])
    ).hexdigest()
    binding_path = Path(
        receipt["arena_session_binding_receipt_path"]
    )
    assert hashlib.sha256(binding_path.read_bytes()).hexdigest() == (
        receipt["arena_session_binding_receipt_sha256"]
    )

    transcript_events = [
        json.loads(line)
        for line in Path(spec.host_transcript_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event.get("kind") == "rpc"
        and event.get("phase") == "applied"
        and event.get("op") == "open"
        for event in transcript_events
    )
    assert any(
        event.get("kind") == "rpc"
        and event.get("phase") == "applied"
        and event.get("op") == "close"
        for event in transcript_events
    )
    assert any(
        event.get("kind") == "rpc_delivery"
        and event.get("seq") == 1
        and event.get("delivered") is True
        for event in transcript_events
    )
    assert runtime.arena_session.token not in receipt_raw.decode("utf-8")
    assert runtime.arena_session.token not in Path(
        spec.host_transcript_path
    ).read_text(encoding="utf-8")

    reducer = R.ContiguousCampaignRunner.__new__(
        R.ContiguousCampaignRunner
    )
    reducer._controller_state_canaries = canaries
    assert reducer._validate_host_blocker(spec, first) == first
    admitted = reducer._sanitize_result(spec, collection.result)
    assert admitted.kind == "blocker"
    assert admitted.reason == (
        "host_blocker:arena_parent_terminal_before_target"
    )
    assert admitted.blocker == first

    # A hand-built lookalike is not the trusted ArenaHostResult type.
    with pytest.raises(
        B.ContainerContractError, match="trusted schema"
    ):
        B._ensure_host_blocker_receipt(
            spec=spec,
            arena_host_result=SimpleNamespace(
                **dataclasses.asdict(machine_result)
            ),
            code="arena_parent_terminal_before_target",
            canaries=canaries,
        )

    # An actual observation cannot be replayed across an attempt/frontier.
    foreign_spec = _short_runner_attempt_spec(request)
    with pytest.raises(
        B.ContainerContractError,
        match=(
            "Arena session binding receipt|immutable seed receipt|"
            "identity or accounting"
        ),
    ):
        B._ensure_host_blocker_receipt(
            spec=foreign_spec,
            arena_host_result=machine_result,
            code="arena_parent_terminal_before_target",
            canaries=canaries,
        )
    replayed = reducer._sanitize_result(
        foreign_spec, collection.result
    )
    assert replayed.kind == "infrastructure"
    assert replayed.blocker is None

    proof = adapter.teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    assert proof.no_descendants is True

    # Mutation with an updated public file hash still lacks the live host HMAC.
    mutated_receipt = dict(receipt)
    mutated_host_result = dict(mutated_receipt["arena_host_result"])
    mutated_host_result["parent_terminal"] = False
    mutated_receipt["arena_host_result"] = mutated_host_result
    mutated_receipt["arena_host_result_sha256"] = hashlib.sha256(
        R._canonical_json(mutated_host_result)
    ).hexdigest()
    os.chmod(receipt_path, 0o600)
    receipt_path.write_bytes(B._canonical_json_bytes(mutated_receipt))
    os.chmod(receipt_path, 0o400)
    mutated_evidence = dataclasses.replace(
        first,
        receipt_sha256=hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
    )
    mutated = reducer._sanitize_result(
        spec,
        dataclasses.replace(
            collection.result, blocker=mutated_evidence
        ),
    )
    assert mutated.kind == "infrastructure"
    assert mutated.blocker is None


def test_candidate_manifest_is_bound_to_arena_and_bridge_exports(
    request: pytest.FixtureRequest,
):
    import arc_agi3_arena_rpc as A
    import arc_agi3_codex_app_server_transport as T
    import arc_agi3_contiguous_runner as R

    spec = _short_runner_attempt_spec(request)
    output = Path(spec.output_dir)
    source = output / "source"
    source.mkdir(mode=0o700)
    source_payloads = {
        "legs.py": b"LEGS = ()\n",
        "players.py": b"PLAYERS = ()\n",
        "solve.py": b"def solve(env):\n    return None\n",
    }
    for name, raw in source_payloads.items():
        (source / name).write_bytes(raw)
        (Path(spec.workspace_dir) / name).write_bytes(raw)
    outcome = output / "worker_outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_proposer_worker",
                "attempt_id": spec.attempt_id,
                "authoritative": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path = [1] * 9
    exported_hashes = {
        "worker_outcome.json": hashlib.sha256(
            outcome.read_bytes()
        ).hexdigest(),
        **{
            f"source/{name}": hashlib.sha256(raw).hexdigest()
            for name, raw in source_payloads.items()
        },
    }
    bridge_token = "d" * 64
    B._write_private_token(
        Path(spec.bridge_token_file_path), bridge_token
    )
    boundary_request_body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_request",
        "protocol_version":
            spec.proposer_transport.bridge_protocol_version,
        "attempt_id": spec.attempt_id,
        "request_id": str(uuid.uuid4()),
        "sequence": 2,
        "session_nonce": "1" * 32,
        "operation": "arena_step",
        "mutation_id": f"{spec.attempt_id}:00000001",
        "challenge_nonce": "2" * 32,
        "arguments": {"action": 1},
    }
    boundary_request = {
        **boundary_request_body,
        "auth_hmac": hmac.new(
            bridge_token.encode("ascii"),
            T.canonical_json(boundary_request_body),
            hashlib.sha256,
        ).hexdigest(),
    }
    workspace_inventory = T.inventory_controller_state(
        Path(spec.workspace_dir)
    )
    boundary = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_target_boundary",
        "attempt_id": spec.attempt_id,
        "game": spec.game,
        "target_level": spec.target_level,
        "levels_before": 8,
        "levels_completed": 9,
        "arena_binding_sha256": "e" * 64,
        "bridge_request_id": boundary_request["request_id"],
        "bridge_sequence": 2,
        "bridge_mutation_id": f"{spec.attempt_id}:00000001",
        "crossing_action_sha256": hashlib.sha256(
            T.canonical_json(1)
        ).hexdigest(),
        "exploration_suffix_sha256": hashlib.sha256(
            T.canonical_json([1])
        ).hexdigest(),
        "exploration_suffix_length": 1,
        "workspace_tree_sha256":
            workspace_inventory.tree_sha256,
        "workspace_inventory_sha256":
            workspace_inventory.inventory_sha256,
        "workspace_file_count": workspace_inventory.file_count,
        "workspace_total_bytes": workspace_inventory.total_bytes,
    }
    boundary_result = {
        "target_reached": True,
        "boundary": boundary,
        "boundary_sha256": hashlib.sha256(
            T.canonical_json(boundary)
        ).hexdigest(),
    }
    boundary_response = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_response",
        "attempt_id": spec.attempt_id,
        "request_id": boundary_request["request_id"],
        "sequence": 2,
        "success": True,
        "result": boundary_result,
        "error": None,
    }
    target_boundary = B._freeze_target_boundary_workspace(
        spec=spec,
        request=boundary_request,
        response=boundary_response,
    )
    manifest_path = output / B.CANDIDATE_MANIFEST_NAME
    manifest = {
        "schema": 1,
        "game": spec.game,
        "target_level": spec.target_level,
        "parent_checkpoint_sha256": spec.parent_checkpoint_sha256,
        "target_boundary_sha256":
            target_boundary.boundary_sha256,
        "target_boundary_sequence":
            target_boundary.bridge_sequence,
        "target_boundary_mutation_id":
            target_boundary.bridge_mutation_id,
        "boundary_workspace_tree_sha256":
            target_boundary.workspace_tree_sha256,
        "candidate_path": candidate_path,
        "exported_files_sha256": exported_hashes,
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    request_body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_request",
        "protocol_version":
            spec.proposer_transport.bridge_protocol_version,
        "attempt_id": spec.attempt_id,
        "request_id": str(uuid.uuid4()),
        "sequence": 3,
        "session_nonce": "1" * 32,
        "operation": "candidate_publish",
        "mutation_id": f"{spec.attempt_id}:00000002",
        "challenge_nonce": "2" * 32,
        "arguments": {
            "candidate_path": candidate_path,
            "exports": {name: name for name in source_payloads},
        },
    }
    request_event = {
        **request_body,
        "auth_hmac": hmac.new(
            bridge_token.encode("ascii"),
            T.canonical_json(request_body),
            hashlib.sha256,
        ).hexdigest(),
    }
    response_event = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_response",
        "attempt_id": spec.attempt_id,
        "request_id": request_body["request_id"],
        "sequence": request_body["sequence"],
        "success": True,
        "result": {
            "outcome": "candidate",
            "manifest": B.CANDIDATE_MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
            "exported_files_sha256": exported_hashes,
            "total_export_bytes": sum(
                len(raw) for raw in source_payloads.values()
            ),
        },
        "error": None,
    }
    arena_result = A.ArenaHostResult(
        binding_sha256="e" * 64,
        game=spec.game,
        exploration_mode="continue_parent",
        parent_level=8,
        levels_completed=9,
        parent_path=tuple([1] * 8),
        path=tuple(candidate_path),
        parent_replay_steps=8,
        exploration_steps=1,
        resets=0,
        total_steps=9,
        parent_terminal=False,
        parent_snapshot_sha256="f" * 64,
    )
    candidate = R.PromotionCandidate(
        game=spec.game,
        from_level=8,
        to_level=9,
        parent_checkpoint_sha256=spec.parent_checkpoint_sha256,
        candidate_manifest_path=str(manifest_path),
        candidate_manifest_sha256=manifest_sha256,
        probe_isolation_mode=(
            R.Contract.VERIFIED_ISOLATED_CLONE_MODE
        ),
        probe_isolation_evidence_sha256="9" * 64,
        supervisory_handoff_sha256=None,
        supervisory_native_reproduction_receipt_sha256=None,
    )
    evidence = B._validate_authenticated_candidate_export(
        spec=spec,
        candidate=candidate,
        arena_host_result=arena_result,
        output_root=output,
        bridge_events=(
            ("bridge_request", boundary_request),
            ("bridge_response", boundary_response),
            ("bridge_request", request_event),
            ("bridge_response", response_event),
        ),
        target_boundary=target_boundary,
    )
    assert evidence.normalized_path == tuple(candidate_path)
    assert dict(evidence.exported_files_sha256) == exported_hashes

    substituted = output / "substituted_candidate_path.json"
    substituted.write_bytes(manifest_path.read_bytes())
    substituted_candidate = dataclasses.replace(
        candidate,
        candidate_manifest_path=str(substituted),
    )
    with pytest.raises(
        B.ContainerContractError,
        match="noncanonical manifest",
    ):
        B._validate_authenticated_candidate_export(
            spec=spec,
            candidate=substituted_candidate,
            arena_host_result=arena_result,
            output_root=output,
            bridge_events=(
                ("bridge_request", boundary_request),
                ("bridge_response", boundary_response),
                ("bridge_request", request_event),
                ("bridge_response", response_event),
            ),
            target_boundary=target_boundary,
        )
    substituted.unlink()

    manifest["candidate_path"] = [1] * 8 + [2]
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    wrong_path_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        B.ContainerContractError,
        match="differs from the trusted Arena result",
    ):
        B._validate_authenticated_candidate_export(
            spec=spec,
            candidate=dataclasses.replace(
                candidate,
                candidate_manifest_sha256=wrong_path_sha256,
            ),
            arena_host_result=arena_result,
            output_root=output,
            bridge_events=(
                ("bridge_request", boundary_request),
                ("bridge_response", boundary_response),
                ("bridge_request", request_event),
                (
                    "bridge_response",
                    {
                        **response_event,
                        "result": {
                            **response_event["result"],
                            "manifest_sha256": wrong_path_sha256,
                        },
                    },
                ),
            ),
            target_boundary=target_boundary,
        )


def test_wip_manifest_is_bound_to_authenticated_broad_and_source_trees(
    tmp_path: Path,
) -> None:
    import arc_agi3_contiguous_runner as R
    import arc_agi3_contiguous_supervisor as S

    spec = _runner_attempt_spec(tmp_path)
    output = Path(spec.output_dir)
    token = "7" * 64
    token_path = Path(spec.bridge_token_file_path)
    token_path.write_text(token + "\n", encoding="ascii")
    token_path.chmod(0o600)
    payloads = {
        "wip/solver_source/legs.py": b"LEGS = ()\n",
        "wip/solver_source/players.py": b"PLAYERS = ()\n",
        "wip/solver_source/solve.py":
            b"def solve(env):\n    return None\n",
        "wip/context/notes.txt": b"exact retained context\n",
    }
    outcome_path = output / "worker_outcome.json"
    outcome_path.write_text("{}\n", encoding="utf-8")
    for relative, raw in payloads.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    exported_hashes = {
        "worker_outcome.json":
            hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
        **{
            relative: hashlib.sha256(raw).hexdigest()
            for relative, raw in payloads.items()
        },
    }
    wip_root = output / "wip"
    source_root = wip_root / "solver_source"
    manifest = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_wip",
        "game": spec.game,
        "target_level": spec.target_level,
        "frontier_sha256": spec.frontier_sha256,
        "parent_checkpoint_sha256": spec.parent_checkpoint_sha256,
        "wip_root_relative_path": "wip",
        "wip_tree_sha256": S._tree_hash(wip_root),
        "solver_source_relative_path": "wip/solver_source",
        "solver_source_tree_sha256": S._tree_hash(source_root),
        "exported_files_sha256": exported_hashes,
    }
    manifest_path = output / B.WIP_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    request_body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_request",
        "protocol_version":
            spec.proposer_transport.bridge_protocol_version,
        "attempt_id": spec.attempt_id,
        "request_id": str(uuid.uuid4()),
        "sequence": 1,
        "session_nonce": "8" * 32,
        "operation": "wip_publish",
        "mutation_id": f"{spec.attempt_id}:00000001",
        "challenge_nonce": "9" * 32,
        "arguments": {
            "exports": {
                relative.removeprefix("wip/"): relative
                for relative in payloads
            },
        },
    }
    request_event = {
        **request_body,
        "auth_hmac": hmac.new(
            token.encode("ascii"),
            T.canonical_json(request_body),
            hashlib.sha256,
        ).hexdigest(),
    }
    response_event = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_bridge_response",
        "attempt_id": spec.attempt_id,
        "request_id": request_body["request_id"],
        "sequence": request_body["sequence"],
        "success": True,
        "result": {
            "outcome": "wip",
            "manifest": B.WIP_MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
            "exported_files_sha256": exported_hashes,
            "total_export_bytes": sum(
                len(raw) for raw in payloads.values()
            ),
        },
        "error": None,
    }
    evidence = B._validate_authenticated_wip_export(
        spec=spec,
        output_root=output,
        bridge_events=(
            ("bridge_request", request_event),
            ("bridge_response", response_event),
        ),
    )
    assert evidence.wip_tree_sha256 == manifest["wip_tree_sha256"]
    assert (
        evidence.solver_source_tree_sha256
        == manifest["solver_source_tree_sha256"]
    )

    (wip_root / "context" / "notes.txt").write_text(
        "mutated\n", encoding="utf-8"
    )
    with pytest.raises(
        B.ContainerContractError,
        match="bridge digest",
    ):
        B._validate_authenticated_wip_export(
            spec=spec,
            output_root=output,
            bridge_events=(
                ("bridge_request", request_event),
                ("bridge_response", response_event),
            ),
        )

    # Even a fully rehashed, bridge-acknowledged WIP cannot retain source
    # that depends on an ambient game/environment module.
    (wip_root / "context" / "notes.txt").write_bytes(
        payloads["wip/context/notes.txt"]
    )
    bad_players = b"import environment_files\nPLAYERS = ()\n"
    players_relative = "wip/solver_source/players.py"
    (output / players_relative).write_bytes(bad_players)
    payloads[players_relative] = bad_players
    exported_hashes[players_relative] = hashlib.sha256(
        bad_players
    ).hexdigest()
    manifest["exported_files_sha256"] = dict(exported_hashes)
    manifest["wip_tree_sha256"] = S._tree_hash(wip_root)
    manifest["solver_source_tree_sha256"] = S._tree_hash(source_root)
    manifest_path.write_text(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    bad_manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    bad_response = {
        **response_event,
        "result": {
            **response_event["result"],
            "manifest_sha256": bad_manifest_sha256,
            "exported_files_sha256": dict(exported_hashes),
            "total_export_bytes": sum(
                len(raw) for raw in payloads.values()
            ),
        },
    }
    with pytest.raises(
        B.ContainerContractError,
        match="closed source schema",
    ):
        B._validate_authenticated_wip_export(
            spec=spec,
            output_root=output,
            bridge_events=(
                ("bridge_request", request_event),
                ("bridge_response", bad_response),
            ),
        )


def test_orphan_process_pid_reuse_never_authorizes_a_signal(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    import arc_agi3_contiguous_runner as R

    spec = _short_runner_attempt_spec(request)
    pid = 4242
    recorded_identity = "a" * 64
    B._write_private_json_new(
        Path(spec.app_server_control_dir) / "process_start.json",
        {
            "schema": 1,
            "kind": "contiguous_app_server_process_start",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "attempt_spec_sha256":
                R.proposer_attempt_binding_sha256(spec),
            "pid": pid,
            "process_group_id": pid,
            "process_start_identity": recorded_identity,
            "codex_binary_sha256":
                spec.proposer_transport.codex_binary_sha256,
            "codex_binary_bytes":
                spec.proposer_transport.codex_binary_bytes,
            "hard_safety_seconds": spec.hard_safety_seconds,
            "max_auth_refreshes": spec.max_auth_refreshes,
            "state_root": spec.app_server_state_dir,
        },
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(B, "_host_process_absent", lambda _pid: False)
    monkeypatch.setattr(
        B, "_host_process_group_absent", lambda _pgid: False
    )
    monkeypatch.setattr(B.os, "getpgid", lambda _pid: pid)
    monkeypatch.setattr(
        B.probe_transport,
        "observe_os_process_start_identity",
        lambda _pid: "b" * 64,
    )
    monkeypatch.setattr(
        B.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    with pytest.raises(
        B.ContainerContractError,
        match="reused or supervisor process identity",
    ):
        B._reconcile_orphan_app_server_process(spec)
    assert signals == []


def test_prepare_process_death_preserves_staging_then_quarantines_attempt(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R

    class SimulatedProcessDeath(BaseException):
        pass

    spec = _short_runner_attempt_spec(request)
    adapter, low_backend = _prepare_only_adapter(spec)
    original_checkpoint = (
        B.compatibility_closure._publication_checkpoint
    )

    def die_after_partial_staging(checkpoint: str) -> None:
        if checkpoint == "content_manifest_fsynced":
            raise SimulatedProcessDeath(checkpoint)

    monkeypatch.setattr(
        B.compatibility_closure,
        "_publication_checkpoint",
        die_after_partial_staging,
    )
    with pytest.raises(
        SimulatedProcessDeath, match="content_manifest_fsynced"
    ):
        adapter.prepare(spec)
    staging = B._compatibility_staging_path(spec)
    assert staging.is_dir()
    retained_observation = (
        B.compatibility_closure.observe_quarantined_staging(
            B._compatibility_closure_paths(spec)[0]
        )
    )
    assert not B._preparation_quarantine_path(spec).exists()

    monkeypatch.setattr(
        B.compatibility_closure,
        "_publication_checkpoint",
        original_checkpoint,
    )
    restarted = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args, **_kwargs: None,
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=adapter._arena_session_factory,
    )
    with pytest.raises(
        R.BackendPreparationQuarantinedError
    ) as first:
        restarted.prepare(spec)
    receipt_path = Path(
        first.value.quarantine_receipt_path
    )
    receipt = B._read_unaliased_json(
        receipt_path,
        label="test preparation quarantine receipt",
    )
    assert receipt["failure_type"] == (
        "CompatibilityStagingAmbiguityError"
    )
    assert receipt["staging_observation"] == retained_observation
    assert receipt["staging_observation_sha256"] == (
        retained_observation["observation_sha256"]
    )
    assert receipt["old_evidence_reuse_authority"] is False
    assert receipt["fresh_attempt_generation_required"] is True
    assert staging.is_dir()
    with pytest.raises(
        R.BackendPreparationQuarantinedError
    ) as reopened:
        restarted.prepare(spec)
    assert (
        reopened.value.quarantine_receipt_sha256
        == first.value.quarantine_receipt_sha256
    )
    assert staging.is_dir()


def test_ordinary_closure_error_without_retained_staging_remains_retryable(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    spec = _short_runner_attempt_spec(request)
    adapter, _low_backend = _prepare_only_adapter(spec)

    def ordinary_failure(_root):
        raise B.compatibility_closure.CompatibilityClosureError(
            "injected ordinary failure"
        )

    monkeypatch.setattr(
        B.compatibility_closure,
        "prepare_closure",
        ordinary_failure,
    )
    with pytest.raises(
        B.ContainerContractError,
        match="without typed retained-staging ambiguity",
    ):
        adapter.prepare(spec)
    closure_root, _turn_receipt = B._compatibility_closure_paths(spec)
    assert not closure_root.exists()
    assert not B._compatibility_staging_path(spec).exists()
    assert not B._preparation_quarantine_path(spec).exists()


def test_typed_adapter_full_mocked_lifecycle_and_token_secrecy(
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R
    from arc_agi3_arena_rpc import ArenaHostSession

    # Keep the complete generation path below the portable sockaddr_un bound.
    spec = _short_runner_attempt_spec(request)
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)
    observations: list[tuple[Any, Any, Any]] = []

    def collect_result(
        runner_spec, terminal, arena_result, worker_outcome, output_root
    ):
        observations.append((terminal, arena_result, worker_outcome))
        return R.AttemptResult(
            kind="clean_no_progress",
            reason="trusted host observed no completed level",
        )

    adapter = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=collect_result,
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=lambda game, *, binding, parent_path, token:
        ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        ),
    )
    prepared = adapter.prepare(spec)
    token_path = Path(spec.arena_token_file_path)
    token = token_path.read_text(encoding="ascii")
    closure = B.compatibility_closure.validate_closure(
        Path(prepared.compatibility_closure_path),
        prepared.compatibility_closure_receipt_sha256,
    )
    assert closure["status"] == "PASS"
    assert closure["launch_authorized"] is False
    compatibility_turn = B._read_unaliased_json(
        Path(prepared.compatibility_turn_receipt_path),
        label="test compatibility turn receipt",
    )
    assert compatibility_turn["closure"]["client_sha256"] == dict(
        low_backend.inspect_image(spec.image_reference).worker_control_sha256
    )[B.LABEL_ARENA_RPC_CLIENT_SHA256]
    assert compatibility_turn["container"]["container_id"] == (
        adapter._attempts[spec.attempt_id].attestation.container_id
    )
    assert compatibility_turn["authority"] == {
        "scheduler_authority": False,
        "mutation_authority": False,
        "promotion_authority": False,
        "launch_authority": False,
        "runner_reopen_required_before_launch": True,
    }
    docker_runner.log_stdout = f"solver said {token}\\n"
    attestation_bytes = Path(
        prepared.launch_attestation_path
    ).read_bytes()
    assert token.encode("ascii") not in attestation_bytes
    assert token not in json.dumps(dataclasses.asdict(prepared))
    assert token not in json.dumps(compatibility_turn)
    assert adapter.prepare(spec) == prepared

    turn_path = Path(prepared.compatibility_turn_receipt_path)
    retained_turn_raw = turn_path.read_bytes()
    substituted_turn = json.loads(retained_turn_raw)
    substituted_turn["authority"]["launch_authority"] = True
    turn_path.write_bytes(B._canonical_json_bytes(substituted_turn))
    with pytest.raises(
        B.ContainerContractError,
        match="existing .* receipt differs",
    ):
        adapter.launch(spec, prepared)
    turn_path.write_bytes(retained_turn_raw)

    launched = adapter.launch(spec, prepared)
    assert adapter.launch(spec, prepared) == launched
    running = adapter.poll(
        spec=spec,
        prepared=prepared,
        launched=launched,
        timeout_seconds=1,
    )
    assert running.status == "running"

    Path(spec.output_dir, "worker_outcome.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_proposer_worker",
                "attempt_id": spec.attempt_id,
                "authoritative": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    docker_runner.running = False
    terminal = adapter.poll(
        spec=spec,
        prepared=prepared,
        launched=launched,
        timeout_seconds=1,
    )
    assert terminal.status == "exited"
    collection = adapter.collect(
        spec=spec,
        prepared=prepared,
        launched=launched,
        terminal=terminal,
    )
    assert collection.result.kind == "clean_no_progress"
    assert observations[0][1].levels_completed == 8
    assert observations[0][2]["authoritative"] is False
    assert token not in Path(spec.host_transcript_path).read_text(
        encoding="utf-8"
    )
    assert "[REDACTED_RPC_TOKEN]" in Path(
        spec.host_transcript_path
    ).read_text(encoding="utf-8")
    assert adapter.collect(
        spec=spec,
        prepared=prepared,
        launched=launched,
        terminal=terminal,
    ) == collection

    proof = adapter.teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    assert proof.no_descendants is True
    assert proof.container_inspect_absent is True
    assert not token_path.exists()
    assert not Path(spec.arena_socket_path).exists()
    assert adapter.teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    ) == proof
    assert not any(
        command[:3] == ("docker", "container", "kill")
        for command in docker_runner.commands
    )


def test_prewrite_boundary_rejection_is_backend_taint_and_noncounting_scheduler_transition(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R
    from arc_agi3_arena_rpc import ArenaHostSession

    boundary_arguments = {
        "path": "probe.py",
        "text": (
            "from pathlib import Path\n"
            "Path('/private/controller/secret').read_text()\n"
        ),
    }

    # Production BridgeClient rejects the source before it emits a bridge
    # request.  The app-server protocol still retains the paired dynamic-tool
    # attempt, which is the immutable classification authority exercised below.
    callbacks: list[tuple[str, dict[str, Any]]] = []
    prewrite_client = T.BridgeClient.__new__(T.BridgeClient)
    prewrite_client._callback = (
        lambda kind, payload: callbacks.append((kind, payload))
    )
    with pytest.raises(
        T.AppServerTransportError,
        match="clean-room filesystem boundary",
    ):
        prewrite_client.call(
            "workspace_write",
            boundary_arguments,
            idempotency_key="boundary-write-call",
        )
    assert callbacks == []

    spec = _short_runner_attempt_spec(request)
    canary_prefix = "boundary-prewrite-e2e"
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(
            spec, canary_prefix
        ),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)
    adapter_kwargs = _authenticated_adapter_test_kwargs(
        low_backend,
        spec,
        canary_prefix=canary_prefix,
    )
    adapter_kwargs["controller_factory"] = lambda **kwargs: (
        _BoundaryRejectedControllerDouble(
            kwargs["probe_spec"],
            boundary_arguments=boundary_arguments,
            **kwargs,
        )
    )
    adapter = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="clean_no_progress",
            reason="collector observed no authenticated candidate",
        ),
        **adapter_kwargs,
        arena_session_factory=lambda game, *, binding, parent_path, token:
            ArenaHostSession(
                game,
                binding=binding,
                parent_path=parent_path,
                token=token,
                arena_factory=lambda _game: _TinyArena(),
            ),
    )

    prepared = adapter.prepare(spec)
    launched = adapter.launch(spec, prepared)
    Path(spec.output_dir, "worker_outcome.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_proposer_worker",
                "attempt_id": spec.attempt_id,
                "authoritative": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    docker_runner.running = False
    terminal = adapter.poll(
        spec=spec,
        prepared=prepared,
        launched=launched,
        timeout_seconds=1,
    )
    collection = adapter.collect(
        spec=spec,
        prepared=prepared,
        launched=launched,
        terminal=terminal,
    )

    assert collection.result.kind == "tainted"
    assert collection.result.candidate is None
    assert collection.result.wip is None
    assert collection.result.blocker is None
    assert "filesystem_boundary:absolute_path" in collection.result.reason
    taint_receipt = json.loads(
        Path(collection.taint_scan_receipt_path).read_text(
            encoding="utf-8"
        )
    )
    assert taint_receipt["status"] == "TAINT"
    assert "filesystem_boundary:absolute_path" in taint_receipt["hits"]
    export_receipt = json.loads(
        Path(collection.bridge_export_receipt_path).read_text(
            encoding="utf-8"
        )
    )
    assert export_receipt["outcome"] == "tainted"
    assert export_receipt["candidate_manifest_sha256"] is None
    assert export_receipt["wip_manifest_sha256"] is None

    # Reopen the backend evidence through the runner's independent validator,
    # then apply the exact scheduler transition consumed by its journal reducer.
    reducer = R.ContiguousCampaignRunner.__new__(
        R.ContiguousCampaignRunner
    )
    reducer._controller_state_canaries = _test_controller_canaries(
        canary_prefix
    )
    reducer._secret_sentinels = tuple(
        adapter_kwargs["credentials"].leak_sentinels
    )
    reducer._validate_collection(
        spec,
        prepared,
        launched,
        collection,
    )
    transition = R.Scheduler.terminal_policy_transition(
        collection.result.kind
    )
    assert transition.next_lane_phase == "READY"
    assert transition.retry_coordinate_delta == 0
    assert transition.current_attempt_wip_disposition == "discard"
    assert transition.prior_wip_disposition == (
        "revoke_same_thread_frontier_context"
    )
    prior_lineage = object()
    assert R.Scheduler.reduce_terminal_wip(
        transition=transition,
        prior_wip=prior_lineage,
        current_attempt_wip=collection.result.wip,
        exposure_detected=False,
    ) is None
    assert R.advance_exact_frontier_clean_no_progress(
        4, collection.result.kind
    ) == 4

    proof = adapter.teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    assert proof.no_descendants is True


def test_adapter_prepare_failure_cleans_private_rpc_endpoints(
    request: pytest.FixtureRequest,
):
    from arc_agi3_arena_rpc import ArenaHostSession

    spec = _short_runner_attempt_spec(request)
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    docker_runner = FakeDockerRunner(
        low_spec,
        image_records=[
            _image_record(manifest=MANIFEST_A, image_id=IMAGE_ID_A),
            _image_record(manifest=MANIFEST_B, image_id=IMAGE_ID_B),
        ],
    )
    low_backend = B.DockerContainerBackend(docker_runner)
    adapter = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: None,
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=lambda game, *, binding, parent_path, token:
        ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        ),
    )
    with pytest.raises(B.ContainerContractError, match="image digest"):
        adapter.prepare(spec)
    for path in (
        spec.arena_socket_path,
        spec.arena_token_file_path,
        spec.host_transcript_path,
        str(Path(spec.host_transcript_path).parent / "launch_attestation.json"),
    ):
        assert not Path(path).exists()
    assert not any(
        command[:3] == ("docker", "container", "create")
        for command in docker_runner.commands
    )


def test_adapter_rehydrates_fresh_instance_after_supervisor_crash(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R
    from arc_agi3_arena_rpc import ArenaHostSession, ArenaRpcClient

    spec = _short_runner_attempt_spec(request)
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)

    def session_factory(game, *, binding, parent_path, token):
        return ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        )

    first = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=session_factory,
    )
    prepared = first.prepare(spec)

    recovered = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=session_factory,
    )
    # A fresh adapter reopens the durable prepared identity.  It must not
    # silently restart an ownerless controller/container launch epoch.
    assert recovered.prepare(spec) == prepared
    with pytest.raises(
        B.ContainerContractError, match="ownerless|ownership was lost"
    ):
        recovered.launch(spec, prepared)
    assert sum(
        command[:3] == ("docker", "container", "start")
        for command in docker_runner.commands
    ) == 0

    tamper_cases = {
        "missing": lambda values: values.pop(
            B.LABEL_PROPOSER_WORKER_SHA256
        ),
        "extra": lambda values: values.__setitem__(
            "org.gkm.arc.worker.unreviewed.sha256", "9" * 64
        ),
        "renamed": lambda values: values.__setitem__(
            "org.gkm.arc.worker.proposer-renamed.sha256",
            values.pop(B.LABEL_PROPOSER_WORKER_SHA256),
        ),
        "tampered": lambda values: values.__setitem__(
            B.LABEL_PROPOSER_WORKER_SHA256, "0" * 64
        ),
    }
    for label, mutate in tamper_cases.items():
        case_spec = _short_runner_attempt_spec(request)
        case_low = B.validate_runner_attempt_contract(
            case_spec,
            containment_canary_anchor=_test_canary_anchor(
                case_spec, f"worker-hash-{label}"
            ),
        )
        case_runner = FakeDockerRunner(case_low)
        case_backend = B.DockerContainerBackend(case_runner)
        case_original = B.ContiguousDockerAttemptBackend(
            case_backend,
            result_collector=lambda *_args: R.AttemptResult(
                kind="infrastructure"
            ),
            **_authenticated_adapter_test_kwargs(
                case_backend,
                case_spec,
                canary_prefix=f"worker-hash-{label}",
            ),
            arena_session_factory=session_factory,
        )
        case_prepared = case_original.prepare(case_spec)
        attestation_path = Path(
            case_prepared.launch_attestation_path
        )
        attestation = json.loads(attestation_path.read_bytes())
        worker_hashes = attestation["image"]["worker_control_sha256"]
        mutate(worker_hashes)
        os.chmod(attestation_path, 0o600, follow_symlinks=False)
        attestation_path.write_bytes(B._canonical_json_bytes(attestation))
        os.chmod(attestation_path, 0o400, follow_symlinks=False)
        case_recovered = B.ContiguousDockerAttemptBackend(
            case_backend,
            result_collector=lambda *_args: R.AttemptResult(
                kind="infrastructure"
            ),
            **_authenticated_adapter_test_kwargs(
                case_backend,
                case_spec,
                canary_prefix=f"worker-hash-{label}",
            ),
            arena_session_factory=session_factory,
        )
        with pytest.raises(
            B.ContainerContractError,
            match="worker-control hashes",
        ):
            case_recovered.prepare(case_spec)
        case_token = Path(
            case_spec.arena_token_file_path
        ).read_text(encoding="ascii")
        ArenaRpcClient(
            case_spec.arena_socket_path, case_token
        ).close()
        case_original._attempts[
            case_spec.attempt_id
        ].arena_server.wait(timeout=5)


def test_adapter_rehydration_rejects_ambiguous_or_tampered_identity(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R
    from arc_agi3_arena_rpc import ArenaHostSession, ArenaRpcClient

    def session_factory(game, *, binding, parent_path, token):
        return ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        )

    ambiguous_spec = _short_runner_attempt_spec(request)
    ambiguous_low = B.validate_runner_attempt_contract(
        ambiguous_spec,
        containment_canary_anchor=_test_canary_anchor(
            ambiguous_spec, "ambiguous"
        ),
    )
    ambiguous_runner = FakeDockerRunner(ambiguous_low)
    ambiguous_backend = B.DockerContainerBackend(ambiguous_runner)
    original = B.ContiguousDockerAttemptBackend(
        ambiguous_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(
            ambiguous_backend,
            ambiguous_spec,
            canary_prefix="ambiguous",
        ),
        arena_session_factory=session_factory,
    )
    ambiguous_prepared = original.prepare(ambiguous_spec)
    ambiguous_runner.label_query_output = (
        CONTAINER_ID + "\n" + "4" * 64 + "\n"
    )
    fresh = B.ContiguousDockerAttemptBackend(
        ambiguous_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(
            ambiguous_backend,
            ambiguous_spec,
            canary_prefix="ambiguous",
        ),
        arena_session_factory=session_factory,
    )
    with pytest.raises(B.ContainerContractError, match="exactly one"):
        fresh.prepare(ambiguous_spec)

    tampered_spec = _short_runner_attempt_spec(request)
    tampered_low = B.validate_runner_attempt_contract(
        tampered_spec,
        containment_canary_anchor=_test_canary_anchor(
            tampered_spec, "tampered"
        ),
    )
    tampered_runner = FakeDockerRunner(tampered_low)
    tampered_backend = B.DockerContainerBackend(tampered_runner)
    original = B.ContiguousDockerAttemptBackend(
        tampered_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(
            tampered_backend,
            tampered_spec,
            canary_prefix="tampered",
        ),
        arena_session_factory=session_factory,
    )
    tampered_prepared = original.prepare(tampered_spec)

    def corrupt_identity(record):
        record["Config"]["Labels"][B.LABEL_ATTEMPT] = str(uuid.uuid4())

    tampered_runner.container_mutator = corrupt_identity
    fresh = B.ContiguousDockerAttemptBackend(
        tampered_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(
            tampered_backend,
            tampered_spec,
            canary_prefix="tampered",
        ),
        arena_session_factory=session_factory,
    )
    with pytest.raises(B.ContainerContractError, match="identity label"):
        fresh.prepare(tampered_spec)


def test_adapter_rehydrates_live_orphan_as_containment_fault(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R
    from arc_agi3_arena_rpc import ArenaHostSession

    spec = _short_runner_attempt_spec(request)
    low_spec = B.validate_runner_attempt_contract(
        spec,
        containment_canary_anchor=_test_canary_anchor(spec),
    )
    docker_runner = FakeDockerRunner(low_spec)
    low_backend = B.DockerContainerBackend(docker_runner)

    def session_factory(game, *, binding, parent_path, token):
        return ArenaHostSession(
            game,
            binding=binding,
            parent_path=parent_path,
            token=token,
            arena_factory=lambda _game: _TinyArena(),
        )

    first = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure"
        ),
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=session_factory,
    )
    prepared = first.prepare(spec)
    launched = first.launch(spec, prepared)
    # The authenticated bridge double performs the proposer-side orderly
    # Arena close during launch; the host must not impersonate a second client.
    first._attempts[spec.attempt_id].arena_server.wait(timeout=5)

    recovered = B.ContiguousDockerAttemptBackend(
        low_backend,
        result_collector=lambda *_args: R.AttemptResult(
            kind="infrastructure",
            reason="supervisor ownership was lost while worker remained live",
        ),
        **_authenticated_adapter_test_kwargs(low_backend, spec),
        arena_session_factory=session_factory,
    )
    terminal = recovered.poll(
        spec=spec,
        prepared=prepared,
        launched=launched,
        timeout_seconds=1,
    )
    assert terminal.status == "containment_fault"
    assert docker_runner.running is False
    assert any(
        command[:3] == ("docker", "container", "stop")
        for command in docker_runner.commands
    )
    with pytest.raises(
        B.ContainerContractError,
        match="ownerless app-server collection",
    ):
        recovered.collect(
            spec=spec,
            prepared=prepared,
            launched=launched,
            terminal=terminal,
        )
    with pytest.raises(
        B.ContainerContractError,
        match="lacks exact app-server containment evidence",
    ):
        recovered.teardown(
            spec=spec,
            prepared=prepared,
            launched=launched,
            cause="containment_fault",
        )
    assert sum(
        command[:3] == ("docker", "container", "start")
        for command in docker_runner.commands
    ) == 1


@pytest.mark.parametrize("surface", ("workspace", "app_server_state"))
def test_fresh_contract_rejects_mutable_tree_drift(
    request: pytest.FixtureRequest,
    surface: str,
):
    spec = _short_runner_attempt_spec(request)
    root = Path(
        spec.workspace_dir
        if surface == "workspace"
        else spec.app_server_state_dir
    )
    (root / "unexpected-post-snapshot").write_text(
        "drift\n", encoding="utf-8"
    )
    with pytest.raises(
        B.ContainerContractError,
        match="workspace, or staged state hash changed",
    ):
        B.validate_runner_attempt_contract(
            spec,
            containment_canary_anchor=_test_canary_anchor(spec),
        )


def test_adapter_recovers_unacknowledged_teardown_from_intent(
    request: pytest.FixtureRequest,
):
    import arc_agi3_contiguous_runner as R

    spec = _short_runner_attempt_spec(request)
    host_root = Path(spec.host_transcript_path).parent
    fixture_root = (
        Path(spec.generation_dir).parent.parent
        / "host_canary_fixture"
    )
    fixture_root.mkdir(mode=0o700)
    roots = {}
    for name in ("repository", "home", "control", "sibling"):
        roots[name] = fixture_root / name
        roots[name].mkdir(mode=0o700)
    credential_path = fixture_root / "auth" / "auth.json"
    credential_path.parent.mkdir(mode=0o700)
    credential_path.write_text('{"token":"unchanged"}\n')
    credential_path.chmod(0o600)
    environment: dict[str, str] = {}

    def new_operator():
        return CanaryOperator.HostContainmentCanaryOperator(
            repository_root=roots["repository"],
            home_root=roots["home"],
            credential_source_path=credential_path,
            controller_control_root=roots["control"],
            sibling_lane_root=roots["sibling"],
            environment=environment,
        )

    planting_authority = B.ContiguousDockerAttemptBackend.__new__(
        B.ContiguousDockerAttemptBackend
    )
    planting_authority._controller_state_canaries = ()
    planting_authority._canary_operator = new_operator()
    planting_authority._attempt_controller_canaries = {}
    planting_authority._attempt_canary_plantings = {}
    anchor = planting_authority._ensure_canary_anchor(spec)

    # The probe adapter is genuine, but this recovery path only needs its
    # attempt-wide Docker identity reconciliation.
    docker_runner = FakeDockerRunner(
        B.validate_runner_attempt_contract(
            spec,
            containment_canary_anchor=anchor,
        )
    )
    low_backend = B.DockerContainerBackend(docker_runner)
    probe_executor = B.DockerWorkspaceProbeExecutor.__new__(
        B.DockerWorkspaceProbeExecutor
    )
    probe_executor._backend = low_backend

    class RecoveryControllerLauncher:
        def recover_absence(
            self,
            *,
            binding,
            controller_container_id,
            egress_proxy_container_id,
            controller_launch_receipt_sha256,
            guardian_start_receipt_sha256,
        ):
            body = {
                "schema": 1,
                "kind": "arc_agi3_controller_absence",
                "campaign_id": binding.campaign_id,
                "generation_id": binding.generation_id,
                "attempt_id": binding.attempt_id,
                "attempt_spec_sha256":
                    binding.attempt_spec_sha256,
                "controller_container_id":
                    controller_container_id,
                "egress_proxy_container_id":
                    egress_proxy_container_id,
                "controller_launch_receipt_sha256":
                    controller_launch_receipt_sha256,
                "guardian_start_receipt_sha256":
                    guardian_start_receipt_sha256,
                "authoritative_identity":
                    "controller_container_cgroup",
                "controller_inspect_absent": True,
                "controller_identity_query_empty": True,
                "controller_top_absent": True,
                "controller_no_descendants": True,
                "egress_proxy_inspect_absent": True,
                "egress_proxy_identity_query_empty": True,
                "egress_proxy_top_absent": True,
                "egress_proxy_no_descendants": True,
            }
            path = host_root / "controller_absence_receipt.json"
            if not path.exists():
                B._write_private_json_new(path, body)
            else:
                assert B._read_unaliased_json(
                    path, label="test controller absence"
                ) == body
            digest = B._hash_unaliased_regular_file(
                path, label="test controller absence"
            )
            return T.ControllerContainerStop(
                controller_container_id=controller_container_id,
                egress_proxy_container_id=egress_proxy_container_id,
                controller_inspect_absent=True,
                controller_identity_query_empty=True,
                controller_top_absent=True,
                controller_no_descendants=True,
                egress_proxy_inspect_absent=True,
                egress_proxy_identity_query_empty=True,
                egress_proxy_top_absent=True,
                egress_proxy_no_descendants=True,
                absence_receipt_path=str(path),
                absence_receipt_sha256=digest,
            )

    credentials = T.ExternalChatGptCredentials(
        access_token="test-access-token",
        account_id="test-account",
        plan_type=None,
        leak_sentinels=("test-access-token",),
        source_path=str(credential_path),
    )

    def new_adapter(operator):
        return B.ContiguousDockerAttemptBackend(
            low_backend,
            result_collector=lambda *_args: R.AttemptResult(
                kind="infrastructure"
            ),
            credentials=credentials,
            probe_executor=probe_executor,
            controller_factory=lambda **_kwargs: None,
            controller_container_launcher=(
                RecoveryControllerLauncher()
            ),
            canary_operator=operator,
            bridge_client_factory=lambda **_kwargs: None,
            arena_session_factory=lambda *_args, **_kwargs: None,
            arena_server_factory=lambda *_args, **_kwargs: None,
            arena_client_factory=lambda *_args, **_kwargs: None,
        )

    original = new_adapter(new_operator())
    assert original._ensure_canary_anchor(spec) == anchor
    prepared = R.BackendPreparation(
        preparation_id="fresh-recovery-preparation",
        launch_attestation_path=str(
            host_root / "launch_attestation.json"
        ),
        launch_attestation_sha256="1" * 64,
        observed_image_digest=spec.image_digest,
        image_observation_sha256="2" * 64,
        container_observation_sha256="3" * 64,
        bridge_policy_receipt_path=spec.bridge_policy_receipt_path,
        bridge_policy_receipt_sha256="4" * 64,
        arena_session_binding_receipt_path=str(
            host_root / "arena_session_binding_receipt.json"
        ),
        arena_session_binding_receipt_sha256="9" * 64,
        compatibility_closure_path=str(
            host_root / B.COMPATIBILITY_CLOSURE_DIRECTORY
        ),
        compatibility_closure_receipt_sha256="b" * 64,
        compatibility_turn_receipt_path=str(
            host_root / B.COMPATIBILITY_TURN_RECEIPT_NAME
        ),
        compatibility_turn_receipt_sha256="c" * 64,
        arena_transport=B.ARENA_VOLUME_TRANSPORT,
        arena_volume_name=B.arena_volume_name(
            B.AttemptIdentity(
                campaign_id=spec.campaign_id,
                generation_id=spec.generation_id,
                attempt_id=spec.attempt_id,
                game=spec.game,
                target_level=spec.target_level,
            )
        ),
        arena_volume_observation_sha256="4" * 64,
        arena_relay_container_id=RELAY_CONTAINER_ID,
        arena_relay_image_digest=RELAY_MANIFEST,
        arena_relay_image_observation_sha256="5" * 64,
        arena_relay_container_observation_sha256="6" * 64,
        arena_relay_readiness_receipt_path=str(
            host_root / "arena_volume_readiness.json"
        ),
        arena_relay_readiness_receipt_sha256="7" * 64,
        arena_relay_attach_argv_sha256="8" * 64,
        arena_relay_socket_identity_sha256="9" * 64,
        arena_relay_preparation_receipt_path=str(
            host_root / "arena_volume_preparation.json"
        ),
        arena_relay_preparation_receipt_sha256="a" * 64,
        probe_isolation_mode=(
            R.Contract.VERIFIED_ISOLATED_CLONE_MODE
        ),
        probe_isolation_evidence_sha256="a" * 64,
        neutral_cwd_attestation_path=str(
            host_root / "neutral_cwd_attestation.json"
        ),
        neutral_cwd_attestation_sha256="5" * 64,
        app_server_config_receipt_path=str(
            host_root / "app_server_config_receipt.json"
        ),
        app_server_config_receipt_sha256="6" * 64,
        codex_binary_receipt_path=str(
            host_root / "codex_binary_receipt.json"
        ),
        codex_binary_receipt_sha256="7" * 64,
        protocol_schema_receipt_path=str(
            host_root / "protocol_schema_receipt.json"
        ),
        protocol_schema_receipt_sha256="8" * 64,
        controller_image_digest=(
            spec.proposer_transport.controller_image_digest
        ),
        controller_egress_proxy_image_digest=(
            spec.proposer_transport
            .controller_egress_proxy_image_digest
        ),
        controller_egress_policy_sha256=(
            spec.proposer_transport
            .controller_egress_policy_sha256
        ),
        controller_canary_escrow_path=anchor.escrow_path,
        controller_canary_escrow_sha256=anchor.escrow_sha256,
        controller_canary_escrow_identity_sha256=(
            anchor.escrow_identity_sha256
        ),
        controller_canary_commitments_json=(
            anchor.commitments_json
        ),
        controller_canary_commitments_sha256=(
            anchor.commitments_sha256
        ),
        controller_canary_placement_descriptors_json=(
            anchor.placement_descriptors_json
        ),
        controller_canary_placement_descriptors_sha256=(
            anchor.placement_descriptors_sha256
        ),
        controller_supply_chain_unobserved_until_launch=True,
    )
    readiness_nonce = "0" * 64
    readiness_path = host_root / "arena_volume_readiness.json"
    B._write_private_json_new(
        readiness_path,
        {
            "schema": 1,
            "kind": "arc_agi3_arena_volume_relay_readiness",
            "status": "READY",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "readiness_nonce": readiness_nonce,
            "relay_pid": 1,
            "socket_path": B.PROPOSER_RPC_SOCKET_DESTINATION,
            "socket_mode": 0o666,
            "network_mode_required": "none",
            "transport": B.ARENA_VOLUME_TRANSPORT,
        },
    )
    readiness_sha256 = B._hash_unaliased_regular_file(
        readiness_path, label="test Arena readiness receipt"
    )
    preparation_path = host_root / "arena_volume_preparation.json"
    B._write_private_json_new(
        preparation_path,
        {
            "schema": 1,
            "kind": "arc_agi3_arena_volume_preparation",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "transport": B.ARENA_VOLUME_TRANSPORT,
            "volume_name": prepared.arena_volume_name,
            "volume_observation_sha256":
                prepared.arena_volume_observation_sha256,
            "relay_container_id":
                prepared.arena_relay_container_id,
            "relay_image_reference":
                spec.proposer_transport.arena_relay_image_reference,
            "relay_image_digest":
                prepared.arena_relay_image_digest,
            "relay_image_observation_sha256":
                prepared.arena_relay_image_observation_sha256,
            "relay_container_observation_sha256":
                prepared.arena_relay_container_observation_sha256,
            "readiness_nonce": readiness_nonce,
            "readiness_receipt_path": str(readiness_path),
            "readiness_receipt_sha256": readiness_sha256,
            "attach_argv_sha256":
                prepared.arena_relay_attach_argv_sha256,
            "arena_socket_identity_sha256":
                prepared.arena_relay_socket_identity_sha256,
        },
    )
    preparation_sha256 = B._hash_unaliased_regular_file(
        preparation_path, label="test Arena preparation receipt"
    )
    prepared = dataclasses.replace(
        prepared,
        arena_relay_readiness_receipt_sha256=readiness_sha256,
        arena_relay_preparation_receipt_sha256=preparation_sha256,
    )
    attachment_receipt = {
        "schema": 1,
        "kind": "arc_agi3_attached_arena_relay",
        "status": "PASS",
        "relay_container_id": RELAY_CONTAINER_ID,
        "threads_stopped": True,
    }
    B._write_private_json_new(
        host_root / "arena_volume_teardown.json",
        {
            "schema": 1,
            "kind": "arc_agi3_arena_volume_teardown",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "transport": B.ARENA_VOLUME_TRANSPORT,
            "preparation_receipt_sha256": preparation_sha256,
            "relay_container_id": RELAY_CONTAINER_ID,
            "volume_name": prepared.arena_volume_name,
            "attachment_status": "CLEAN_EOF",
            "attachment_receipt": attachment_receipt,
            "attachment_receipt_sha256":
                B._json_sha256(attachment_receipt),
            "relay_inspect_absent": True,
            "relay_top_absent": True,
            "relay_identity_query_empty": True,
            "volume_inspect_absent": True,
            "volume_identity_query_empty": True,
        },
    )
    launched = R.BackendLaunch(
        backend_id=CONTAINER_ID,
        container_id=CONTAINER_ID,
        running_observation_sha256="9" * 64,
        substrate_identity_sha256="8" * 64,
        substrate_preflight_receipt_path=str(
            host_root / "substrate_preflight_receipt.json"
        ),
        substrate_preflight_receipt_sha256="7" * 64,
        bridge_runtime_attestation_path=str(
            host_root / "bridge_runtime_attestation.json"
        ),
        bridge_runtime_attestation_sha256="a" * 64,
        app_server_runtime_receipt_path=str(
            host_root / "app_server_runtime_receipt.json"
        ),
        app_server_runtime_receipt_sha256="b" * 64,
        app_server_pid=999_991,
        app_server_process_start="c" * 64,
        app_server_process_group_id=999_991,
        app_server_pid_is_diagnostic=True,
        process_identity_authority="controller_container_cgroup",
        controller_container_id="6" * 64,
        controller_image_digest=prepared.controller_image_digest,
        egress_proxy_container_id="7" * 64,
        egress_proxy_image_digest=(
            prepared.controller_egress_proxy_image_digest
        ),
        egress_policy_sha256=(
            prepared.controller_egress_policy_sha256
        ),
        controller_launch_intent_sha256="d" * 64,
        controller_launch_receipt_path=str(
            host_root / "controller_launch_receipt.json"
        ),
        controller_launch_receipt_sha256="e" * 64,
        controller_guardian_start_receipt_path=str(
            host_root / "controller_guardian_start.json"
        ),
        controller_guardian_start_receipt_sha256="f" * 64,
        controller_supply_chain_manifest_sha256="1" * 64,
        codex_thread_id=str(uuid.uuid4()),
        codex_turn_id=str(uuid.uuid4()),
        thread_binding_path=str(
            host_root / "thread_binding.json"
        ),
        thread_binding_sha256="2" * 64,
        transcript_chain_receipt_path=str(
            host_root / "transcript_chain.json"
        ),
        transcript_chain_receipt_sha256="3" * 64,
        transcript_chain_sha256="4" * 64,
        thread_rebinding_receipt_path=None,
        thread_rebinding_receipt_sha256=None,
    )
    commitments = json.loads(anchor.commitments_json)
    B._write_private_json_new(
        host_root / "controller_state_scan_receipt.json",
        {
            "controller_state_scan": {
                "canary_commitments": commitments,
                "canary_occurrences": 0,
                "status": "CLEAN",
            }
        },
    )
    B._write_private_json_new(
        Path(spec.generation_dir)
        / "retained_canary_scan_receipt.json",
        {
            "retained_canary_scan": {
                "canary_commitments": commitments,
                "canary_occurrences": 0,
                "status": "CLEAN",
            }
        },
    )
    original._ensure_teardown_intent(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    docker_runner.created = True
    docker_runner.started = True
    docker_runner.running = False
    docker_runner.removed = True
    reveal = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_reveals"
        / f"{spec.generation_id}.json"
    )
    cleanup = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_cleanups"
        / f"{spec.generation_id}.json"
    )
    assert not reveal.exists()
    assert not cleanup.exists()

    # A genuinely fresh adapter has no process-local planting cache.  It must
    # reopen escrow/intent, re-prove all three cgroups absent, publish reveal,
    # and finish crash-safe exact-placement cleanup.
    recovered = new_adapter(new_operator())
    proof = recovered.teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    assert proof.no_descendants is True
    assert proof.arena_relay_container_id == RELAY_CONTAINER_ID
    assert proof.arena_volume_name == prepared.arena_volume_name
    assert proof.arena_relay_attachment_status == "CLEAN_EOF"
    assert proof.arena_relay_inspect_absent is True
    assert proof.arena_relay_top_absent is True
    assert proof.arena_relay_identity_query_empty is True
    assert proof.arena_volume_inspect_absent is True
    assert proof.arena_volume_identity_query_empty is True
    assert Path(
        proof.arena_relay_teardown_receipt_path
    ) == host_root / "arena_volume_teardown.json"
    assert proof.canary_reveal_path == str(reveal)
    assert proof.canary_cleanup_receipt_path == str(cleanup)
    assert reveal.is_file()
    assert cleanup.is_file()
    assert (host_root / "teardown_proof.json").is_file()
    for item in original._attempt_controller_canaries[spec.attempt_id]:
        if item.category == "environment":
            assert item.location_name not in environment
        else:
            assert not Path(item.location_name).exists()

    # A second fresh adapter must reopen the durable proof and cleanup
    # receipt without republishing or weakening any terminal evidence.
    repeated = new_adapter(new_operator()).teardown(
        spec=spec,
        prepared=prepared,
        launched=launched,
        cause="normal_exit",
    )
    assert repeated == proof


def test_fast_worker_exit_between_start_and_inspect_is_accepted(
    attempt_spec: B.AttemptSpec,
):
    class FastExitRunner(FakeDockerRunner):
        def run(self, argv, *, timeout_seconds=None):
            result = super().run(argv, timeout_seconds=timeout_seconds)
            if tuple(argv)[:3] == ("docker", "container", "start"):
                self.running = False
            return result

    runner = FastExitRunner(attempt_spec)
    backend = B.DockerContainerBackend(runner)
    attestation = backend.build_launch_attestation(attempt_spec)
    started = backend.start_attested(attestation, attempt_spec)
    state = backend.observe_container_state(
        attestation, attempt_spec, timeout_seconds=1
    )
    assert started.attestation == attestation
    assert state.status == "exited"
    assert state.running is False


PROBE_IMAGE_REFERENCE = (
    f"registry.example/gkm/arc-workspace-probe@{MANIFEST_B}"
)
PROBE_CONTAINER_ID = "4" * 64
PARENT_CONTAINER_ID = "5" * 64


def _probe_image_record() -> dict[str, Any]:
    recipe = (
        Path(__file__).parent
        / "container"
        / "Containerfile.arc-agi3-workspace-probe"
    )
    return {
        "Id": IMAGE_ID_B,
        "RepoDigests": [PROBE_IMAGE_REFERENCE],
        "Config": {
            "Env": [
                "PATH=/usr/local/bin:/usr/bin:/bin",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PYTHONDONTWRITEBYTECODE=1",
                "PYTHONUNBUFFERED=1",
                "PYTHON_VERSION=3.12.11",
                "PYTHON_PIP_VERSION=25.1",
                "PYTHON_SETUPTOOLS_VERSION=80.9.0",
                "PYTHON_GET_PIP_URL=https://example.invalid/get-pip.py",
                "PYTHON_GET_PIP_SHA256=" + "4" * 64,
                "PYTHON_SHA256=" + "5" * 64,
                "GPG_KEY=public-build-key",
            ],
            "Entrypoint": [B.PYTHON_ENTRYPOINT, "-I"],
            "User": B.PROBE_USER,
            "WorkingDir": B.PROBE_DESTINATION,
            "Cmd": None,
            "Volumes": None,
            "ExposedPorts": None,
            "Labels": {
                B.PROBE_ROLE_LABEL: B.PROBE_ROLE_VALUE,
                B.PROBE_RECIPE_LABEL: hashlib.sha256(
                    recipe.read_bytes()
                ).hexdigest(),
                B.PROBE_BASE_IMAGE_LABEL: B.PROBE_BASE_IMAGE_DIGEST,
                B.PROBE_CONTENT_POLICY_LABEL:
                    B.PROBE_CONTENT_POLICY_VALUE,
            },
        },
    }


def _probe_parent_record(
    spec: Any,
    *,
    running: bool = True,
    mutate_labels: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, Any]:
    labels = {
        B.LABEL_CAMPAIGN: spec.campaign_id,
        B.LABEL_GENERATION: spec.generation_id,
        B.LABEL_ATTEMPT: spec.attempt_id,
        B.LABEL_GAME: spec.game,
        B.LABEL_LEVEL: str(spec.target_level),
        **B.trusted_worker_hashes(),
    }
    if mutate_labels is not None:
        mutate_labels(labels)
    return {
        "Id": PARENT_CONTAINER_ID,
        "Config": {
            "Image": spec.image_reference,
            "Labels": labels,
        },
        "State": {
            "Status": "running" if running else "exited",
            "Running": running,
            "Paused": False,
            "Restarting": False,
        },
    }


def _probe_turn_launch(
    spec: Any, parent_record: Mapping[str, Any]
) -> Any:
    import arc_agi3_contiguous_runner as R

    thread_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    bridge_hash = "6" * 64
    app_hash = "7" * 64
    chain_hash = "8" * 64
    host_root = Path(spec.host_transcript_path).parent
    binding_path = host_root / "turn_start_binding.json"
    binding = {
        "schema": 1,
        "kind": "contiguous_turn_start_binding",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256": R.proposer_attempt_binding_sha256(spec),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "thread_mode": spec.thread_mode,
        "bridge_runtime_attestation_sha256": bridge_hash,
        "app_server_runtime_receipt_sha256": app_hash,
        "reasoning_effort": spec.effort,
        "model": spec.proposer_transport.model,
        "transcript_chain_sha256": chain_hash,
    }
    binding_path.write_bytes(B._canonical_json_bytes(binding))
    binding_path.chmod(0o400)
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    return R.BackendLaunch(
        backend_id=PARENT_CONTAINER_ID,
        container_id=PARENT_CONTAINER_ID,
        running_observation_sha256=B._json_sha256(parent_record),
        substrate_identity_sha256="5" * 64,
        substrate_preflight_receipt_path=str(
            host_root / "substrate_preflight_receipt.json"
        ),
        substrate_preflight_receipt_sha256="4" * 64,
        bridge_runtime_attestation_path=str(
            host_root / "bridge_runtime_attestation.json"
        ),
        bridge_runtime_attestation_sha256=bridge_hash,
        app_server_runtime_receipt_path=str(
            host_root / "app_server_runtime_receipt.json"
        ),
        app_server_runtime_receipt_sha256=app_hash,
        app_server_pid=12345,
        app_server_process_start="123",
        app_server_process_group_id=12345,
        app_server_pid_is_diagnostic=True,
        process_identity_authority="controller_container_cgroup",
        controller_container_id="a" * 64,
        controller_image_digest=(
            spec.proposer_transport.controller_image_digest
        ),
        egress_proxy_container_id="b" * 64,
        egress_proxy_image_digest=(
            spec.proposer_transport
            .controller_egress_proxy_image_digest
        ),
        egress_policy_sha256=(
            spec.proposer_transport.controller_egress_policy_sha256
        ),
        controller_launch_intent_sha256="c" * 64,
        controller_launch_receipt_path=str(
            host_root / "controller_launch_receipt.json"
        ),
        controller_launch_receipt_sha256="d" * 64,
        controller_guardian_start_receipt_path=str(
            host_root / "controller_guardian_start.json"
        ),
        controller_guardian_start_receipt_sha256="e" * 64,
        controller_supply_chain_manifest_sha256="f" * 64,
        codex_thread_id=thread_id,
        codex_turn_id=turn_id,
        thread_binding_path=str(binding_path),
        thread_binding_sha256=binding_sha256,
        transcript_chain_receipt_path=str(
            host_root / "transcript_chain_receipt.json"
        ),
        transcript_chain_receipt_sha256="9" * 64,
        transcript_chain_sha256=chain_hash,
        thread_rebinding_receipt_path=None,
        thread_rebinding_receipt_sha256=None,
    )


def _chmod_probe_tree_readonly(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            directories.append(path)
        else:
            path.chmod(0o444)
    for path in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        path.chmod(0o555)


def _probe_request(
    spec: Any,
    launched: Any,
    *,
    dynamic_request_id: str | int = "request-1",
    dynamic_call_id: str = "call-1",
) -> T.ProbeExecutionRequest:
    import arc_agi3_contiguous_supervisor as S

    generation = Path(spec.generation_dir)
    request_root = (
        generation / "probe_calls" / str(dynamic_request_id)
    )
    call_root = request_root / dynamic_call_id
    snapshot = call_root / "snapshot"
    (generation / "probe_calls").mkdir(mode=0o700, exist_ok=True)
    (generation / "probe_calls").chmod(0o700)
    request_root.mkdir(mode=0o700, exist_ok=False)
    call_root.mkdir(mode=0o700, exist_ok=False)
    snapshot.mkdir(mode=0o700, exist_ok=False)
    (snapshot / "main.py").write_text(
        "print('probe-ok')\n", encoding="utf-8"
    )
    (snapshot / "payload.txt").write_text(
        "immutable\n", encoding="utf-8"
    )
    _chmod_probe_tree_readonly(snapshot)
    rows = []
    for path in sorted(
        item for item in snapshot.rglob("*") if item.is_file()
    ):
        metadata = path.stat(follow_symlinks=False)
        rows.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": metadata.st_size,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            }
        )
    snapshot_metadata = snapshot.stat(follow_symlinks=False)
    tree_sha256 = S._tree_hash(snapshot)
    manifest = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_workspace_snapshot",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "dynamic_request_id": dynamic_request_id,
        "dynamic_call_id": dynamic_call_id,
        "thread_id": launched.codex_thread_id,
        "turn_id": launched.codex_turn_id,
        "generation_dir": str(generation),
        "call_dir": str(call_root),
        "snapshot_root": str(snapshot),
        "snapshot_device": snapshot_metadata.st_dev,
        "snapshot_inode": snapshot_metadata.st_ino,
        "tree_sha256": tree_sha256,
        "entries": rows,
        "source_workspace_tree_sha256": "a" * 64,
        "no_writeback": True,
    }
    # Exercise the shared parser before the backend independently admits it.
    assert T.workspace_snapshot_manifest_from_dict(manifest)
    manifest_path = call_root / "snapshot_manifest.json"
    manifest_path.write_bytes(T.canonical_json(manifest))
    manifest_path.chmod(0o400)
    return T.ProbeExecutionRequest(
        schema=1,
        campaign_id=spec.campaign_id,
        generation_id=spec.generation_id,
        attempt_id=spec.attempt_id,
        dynamic_request_id=dynamic_request_id,
        dynamic_call_id=dynamic_call_id,
        thread_id=launched.codex_thread_id,
        turn_id=launched.codex_turn_id,
        workspace_snapshot_manifest_path=str(manifest_path),
        workspace_snapshot_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        workspace_snapshot_tree_sha256=tree_sha256,
        entrypoint="main.py",
        arguments=("--mode", "test"),
        timeout_seconds=10,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
        resource_limits=T.ProbeResourceLimits(
            cpus=1.0,
            memory_bytes=64 * 1024 * 1024,
            pids=16,
            tmpfs_bytes=16 * 1024 * 1024,
        ),
        arena_mode="disabled",
        arena_session_id=None,
    )


class FakeProbeDockerRunner:
    def __init__(
        self,
        spec: Any,
        launched: Any,
        request: T.ProbeExecutionRequest,
        *,
        parent_record: Mapping[str, Any],
        image_records: Sequence[dict[str, Any]] | None = None,
        create_hook: Callable[[], None] | None = None,
        create_stdout: str | None = None,
        timed_out: bool = False,
        output_overflow: bool = False,
    ) -> None:
        self.spec = spec
        self.launched = launched
        self.request = request
        self.parent_record = dict(parent_record)
        self.image_records = list(
            image_records or [_probe_image_record()]
        )
        self.image_calls = 0
        self.create_hook = create_hook
        self.create_stdout = create_stdout
        self.timed_out = timed_out
        self.output_overflow = output_overflow
        self.commands: list[tuple[str, ...]] = []
        self.create_command: tuple[str, ...] | None = None
        self.created = False
        self.started = False
        self.removed = False
        self.parent_inspections = 0

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> B.CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        if command[:3] == ("docker", "image", "inspect"):
            index = min(self.image_calls, len(self.image_records) - 1)
            self.image_calls += 1
            return self._result(
                command, stdout=json.dumps([self.image_records[index]])
            )
        if command[:3] == ("docker", "container", "create"):
            self.created = True
            self.create_command = command
            if self.create_hook is not None:
                self.create_hook()
            return self._result(
                command,
                stdout=(
                    self.create_stdout
                    if self.create_stdout is not None
                    else PROBE_CONTAINER_ID + "\n"
                ),
            )
        if command[:3] == ("docker", "container", "inspect"):
            target = command[3]
            if target == PARENT_CONTAINER_ID:
                self.parent_inspections += 1
                return self._result(
                    command, stdout=json.dumps([self.parent_record])
                )
            if target != PROBE_CONTAINER_ID or self.removed:
                return self._result(
                    command, returncode=1, stderr="absent"
                )
            return self._result(
                command,
                stdout=json.dumps([self._probe_container_record()]),
            )
        if command[:3] == ("docker", "container", "ls"):
            output = (
                PROBE_CONTAINER_ID + "\n"
                if self.created and not self.removed
                else ""
            )
            return self._result(command, stdout=output)
        if command[:3] == ("docker", "container", "rm"):
            self.removed = True
            return self._result(command, stdout=PROBE_CONTAINER_ID + "\n")
        if command[:3] == ("docker", "container", "top"):
            return self._result(
                command,
                returncode=1 if self.removed else 0,
                stderr="absent" if self.removed else "",
            )
        raise AssertionError(f"unexpected probe Docker command: {command}")

    def run_attached_stream(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdout_path: Path,
        stderr_path: Path,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> B.AttachedStreamResult:
        command = tuple(argv)
        self.commands.append(command)
        assert command == (
            "docker",
            "container",
            "start",
            "--attach",
            PROBE_CONTAINER_ID,
        )
        self.started = True
        stdout = b"probe-ok\n"
        stderr = b""
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        stdout_path.chmod(0o400)
        stderr_path.chmod(0o400)
        return B.AttachedStreamResult(
            argv=command,
            returncode=(
                None
                if self.timed_out or self.output_overflow
                else 0
            ),
            stdout_path=str(stdout_path),
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stdout_bytes=len(stdout),
            stdout_truncated=self.output_overflow,
            stderr_path=str(stderr_path),
            stderr_sha256=hashlib.sha256(stderr).hexdigest(),
            stderr_bytes=0,
            stderr_truncated=False,
            timed_out=self.timed_out,
            output_overflow=self.output_overflow,
            started_monotonic=1.0,
            finished_monotonic=2.0,
        )

    def _probe_container_record(self) -> dict[str, Any]:
        assert self.create_command is not None
        command = self.create_command
        labels = {
            command[index + 1].split("=", 1)[0]:
                command[index + 1].split("=", 1)[1]
            for index, value in enumerate(command)
            if value == "--label"
        }
        name = command[command.index("--name") + 1]
        image_index = command.index(PROBE_IMAGE_REFERENCE)
        mount_value = command[command.index("--mount") + 1]
        mount_source = mount_value.split("src=", 1)[1].split(",", 1)[0]
        limits = self.request.resource_limits
        running = bool(
            self.started and (self.timed_out or self.output_overflow)
        )
        status = (
            "running"
            if running
            else ("exited" if self.started else "created")
        )
        return {
            "Id": PROBE_CONTAINER_ID,
            "Image": IMAGE_ID_B,
            "Name": "/" + name,
            "Config": {
                "Image": PROBE_IMAGE_REFERENCE,
                "User": B.PROBE_USER,
                "Labels": labels,
                "Env": _probe_image_record()["Config"]["Env"],
                "Cmd": list(command[image_index + 1:]),
                "Entrypoint": [B.PYTHON_ENTRYPOINT],
                "WorkingDir": B.PROBE_DESTINATION,
                "Healthcheck": {"Test": ["NONE"]},
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "NetworkMode": "none",
                "PidMode": "",
                "CgroupnsMode": "private",
                "IpcMode": "private",
                "UTSMode": "",
                "Privileged": False,
                "RestartPolicy": {
                    "Name": "no",
                    "MaximumRetryCount": 0,
                },
                "LogConfig": {"Type": "none", "Config": {}},
                "CapAdd": [],
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges=true"],
                "NanoCpus": int(float(limits.cpus) * 1_000_000_000),
                "Memory": limits.memory_bytes,
                "MemorySwap": limits.memory_bytes,
                "PidsLimit": limits.pids,
                "Devices": [],
                "DeviceRequests": [],
                "Tmpfs": {
                    B.TMPFS_DESTINATION: (
                        "rw,nosuid,nodev,noexec,"
                        f"size={limits.tmpfs_bytes},mode=1777,"
                        "uid=65534,gid=65534"
                    )
                },
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": mount_source,
                    "Destination": B.PROBE_DESTINATION,
                    "RW": False,
                    "Propagation": "rprivate",
                }
            ],
            "NetworkSettings": {"Networks": {}},
            "State": {
                "Status": status,
                "Running": running,
                "Paused": False,
                "Restarting": False,
                "ExitCode": 0,
            },
        }

    @staticmethod
    def _result(
        command: tuple[str, ...],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> B.CommandResult:
        return B.CommandResult(
            argv=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _probe_fixture(
    request: pytest.FixtureRequest,
    *,
    dynamic_request_id: str | int = "request-1",
    dynamic_call_id: str = "call-1",
) -> tuple[Any, Any, T.ProbeExecutionRequest, dict[str, Any]]:
    spec = _short_runner_attempt_spec(request)
    parent = _probe_parent_record(spec)
    launched = _probe_turn_launch(spec, parent)
    probe_request = _probe_request(
        spec,
        launched,
        dynamic_request_id=dynamic_request_id,
        dynamic_call_id=dynamic_call_id,
    )
    return spec, launched, probe_request, parent


def test_workspace_probe_run_crosses_shared_schema_and_exact_teardown(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    runner = FakeProbeDockerRunner(
        spec, launched, probe_request, parent_record=parent
    )
    backend = B.DockerContainerBackend(runner)
    executor = B.DockerWorkspaceProbeExecutor(
        backend, image_reference=PROBE_IMAGE_REFERENCE
    )
    result = executor.run_probe(
        spec=spec, launched=launched, request=probe_request
    )

    assert isinstance(result, T.ProbeExecutionResult)
    assert result.schema == 1
    assert result.request_sha256 == probe_request.sha256()
    assert result.snapshot_tree_sha256 \
        == probe_request.workspace_snapshot_tree_sha256
    assert result.container_absent is True
    assert result.process_group_absent is True
    assert result.descendants_absent is True
    assert result.no_writeback is True
    assert runner.removed is True
    assert runner.parent_inspections == 2
    assert Path(result.stdout_path).read_bytes() == b"probe-ok\n"
    assert Path(result.stderr_path).read_bytes() == b""
    create = runner.create_command
    assert create is not None
    assert "--env" not in create
    assert "--env-file" not in create
    assert ("--network", "none") == (
        create[create.index("--network")],
        create[create.index("--network") + 1],
    )
    assert create[create.index("--log-driver") + 1] == "none"
    assert str(Path(spec.input_dir)) not in create
    assert str(Path(spec.output_dir)) not in create
    assert str(Path(spec.bridge_dir)) not in create
    assert str(Path(spec.arena_token_file_path)) not in create
    containment = json.loads(
        Path(result.containment_attestation_path).read_text(
            encoding="utf-8"
        )
    )
    assert containment["parent_container_id"] == PARENT_CONTAINER_ID
    assert containment["dynamic_call_id"] == "call-1"
    assert containment["snapshot_root_inode"] > 0
    assert containment["call_root_inode"] > 0
    teardown = json.loads(
        Path(result.teardown_receipt_path).read_text(encoding="utf-8")
    )
    assert teardown["identity_query_empty"] is True
    assert teardown["no_writeback"] is True


def test_workspace_probe_rejects_reused_call_and_mount_delimiters(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    runner = FakeProbeDockerRunner(
        spec, launched, probe_request, parent_record=parent
    )
    executor = B.DockerWorkspaceProbeExecutor(
        B.DockerContainerBackend(runner),
        image_reference=PROBE_IMAGE_REFERENCE,
    )
    executor.run_probe(
        spec=spec, launched=launched, request=probe_request
    )
    with pytest.raises(
        B.ContainerContractError, match="fresh exact snapshot"
    ):
        executor.run_probe(
            spec=spec, launched=launched, request=probe_request
        )

    root = _request_private_test_root(
        request, prefix=".a3cb_probe_comma_,"
    )
    comma_spec = _runner_attempt_spec(root)
    comma_parent = _probe_parent_record(comma_spec)
    comma_launch = _probe_turn_launch(comma_spec, comma_parent)
    comma_request = _probe_request(
        comma_spec, comma_launch,
        dynamic_request_id="request-2",
        dynamic_call_id="call-2",
    )
    comma_runner = FakeProbeDockerRunner(
        comma_spec,
        comma_launch,
        comma_request,
        parent_record=comma_parent,
    )
    with pytest.raises(
        B.ContainerContractError, match="mount option parsing"
    ):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(comma_runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=comma_spec,
            launched=comma_launch,
            request=comma_request,
        )
    assert comma_runner.created is False


def test_workspace_probe_snapshot_swap_fails_and_removes_container(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    entrypoint = (
        Path(probe_request.workspace_snapshot_manifest_path).parent
        / "snapshot"
        / "main.py"
    )

    def mutate_snapshot() -> None:
        entrypoint.chmod(0o644)
        entrypoint.write_text("print('swapped')\n", encoding="utf-8")
        entrypoint.chmod(0o444)

    runner = FakeProbeDockerRunner(
        spec,
        launched,
        probe_request,
        parent_record=parent,
        create_hook=mutate_snapshot,
    )
    executor = B.DockerWorkspaceProbeExecutor(
        B.DockerContainerBackend(runner),
        image_reference=PROBE_IMAGE_REFERENCE,
    )
    with pytest.raises(
        B.ContainerContractError,
        match="entries differ|tree hash|snapshot file",
    ):
        executor.run_probe(
            spec=spec, launched=launched, request=probe_request
        )
    assert runner.created is True
    assert runner.removed is True
    assert runner.started is False


@pytest.mark.parametrize(
    "create_stdout",
    ["not-a-container-id\n", "6" * 64 + "\n"],
)
def test_workspace_probe_uncertain_create_ack_reconciles_exact_identity(
    request: pytest.FixtureRequest,
    create_stdout: str,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    runner = FakeProbeDockerRunner(
        spec,
        launched,
        probe_request,
        parent_record=parent,
        create_stdout=create_stdout,
    )
    with pytest.raises(
        B.ContainerContractError,
        match="ambiguous container id|label identity",
    ):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=spec, launched=launched, request=probe_request
        )
    assert runner.created is True
    assert runner.removed is True
    assert runner.started is False
    label_queries = [
        command
        for command in runner.commands
        if command[:3] == ("docker", "container", "ls")
    ]
    assert len(label_queries) >= 3


@pytest.mark.parametrize("mode", ["timeout", "overflow"])
def test_workspace_probe_timeout_or_overflow_proves_absence(
    request: pytest.FixtureRequest,
    mode: str,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    runner = FakeProbeDockerRunner(
        spec,
        launched,
        probe_request,
        parent_record=parent,
        timed_out=mode == "timeout",
        output_overflow=mode == "overflow",
    )
    result = B.DockerWorkspaceProbeExecutor(
        B.DockerContainerBackend(runner),
        image_reference=PROBE_IMAGE_REFERENCE,
    ).run_probe(
        spec=spec, launched=launched, request=probe_request
    )
    assert result.timed_out is (mode == "timeout")
    assert result.output_overflow is (mode == "overflow")
    assert result.exit_code is None
    assert result.container_absent is True
    assert runner.removed is True


def test_workspace_probe_uses_shared_model_visible_timeout_bound(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    over_limit = dataclasses.replace(
        probe_request,
        timeout_seconds=T.MAX_PROBE_TIMEOUT_SECONDS + 1,
    )
    runner = FakeProbeDockerRunner(
        spec, launched, over_limit, parent_record=parent
    )
    with pytest.raises(B.ContainerContractError, match="hard bound"):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=spec,
            launched=launched,
            request=over_limit,
        )
    assert runner.created is False
    assert B.MAX_PROBE_TIMEOUT_SECONDS == T.MAX_PROBE_TIMEOUT_SECONDS


def test_workspace_probe_attempt_reconciliation_is_exact_and_idempotent(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    runner = FakeProbeDockerRunner(
        spec, launched, probe_request, parent_record=parent
    )
    executor = B.DockerWorkspaceProbeExecutor(
        B.DockerContainerBackend(runner),
        image_reference=PROBE_IMAGE_REFERENCE,
    )
    executor.run_probe(
        spec=spec, launched=launched, request=probe_request
    )

    # Model a supervisor crash after a second sibling probe was created but
    # before its per-call terminal receipt could be written.
    runner.created = True
    runner.removed = False
    runner.started = False
    with pytest.raises(
        B.ContainerContractError, match="live workspace probe"
    ):
        executor.prove_attempt_probe_absence(
            spec=spec, stage="pre_collection"
        )

    first = executor.reconcile_attempt_probes(
        spec=spec, stage="startup"
    )
    assert first["removed_container_ids"] == [PROBE_CONTAINER_ID]
    assert first["identity_query_empty"] is True
    assert runner.removed is True
    repeated = executor.reconcile_attempt_probes(
        spec=spec, stage="startup"
    )
    assert repeated == first
    absence = executor.prove_attempt_probe_absence(
        spec=spec, stage="pre_collection"
    )
    assert absence["identity_query_empty"] is True

    # A sibling cannot silently reappear under an already-completed receipt.
    runner.removed = False
    with pytest.raises(B.ContainerContractError, match="reappeared"):
        executor.reconcile_attempt_probes(
            spec=spec, stage="startup"
        )


def test_workspace_probe_rejects_parent_or_image_substitution_before_create(
    request: pytest.FixtureRequest,
):
    spec, launched, probe_request, parent = _probe_fixture(request)
    stale_parent = _probe_parent_record(spec, running=False)
    stale_runner = FakeProbeDockerRunner(
        spec,
        launched,
        probe_request,
        parent_record=stale_parent,
    )
    with pytest.raises(
        B.ContainerContractError, match="proposer.*running|launch observation"
    ):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(stale_runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=spec, launched=launched, request=probe_request
        )
    assert stale_runner.created is False

    after_create_request = _probe_request(
        spec,
        launched,
        dynamic_request_id="request-parent-swap",
        dynamic_call_id="call-parent-swap",
    )
    after_create_runner = FakeProbeDockerRunner(
        spec,
        launched,
        after_create_request,
        parent_record=parent,
    )
    after_create_runner.create_hook = lambda: setattr(
        after_create_runner,
        "parent_record",
        _probe_parent_record(spec, running=False),
    )
    with pytest.raises(
        B.ContainerContractError, match="proposer.*running|launch observation"
    ):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(after_create_runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=spec,
            launched=launched,
            request=after_create_request,
        )
    assert after_create_runner.created is True
    assert after_create_runner.started is False
    assert after_create_runner.removed is True

    # Allocate a distinct call root because the failed attempt retained its
    # request evidence as a scientifically useful failure boundary.
    probe_request_2 = _probe_request(
        spec,
        launched,
        dynamic_request_id="request-image",
        dynamic_call_id="call-image",
    )
    bad_image = _probe_image_record()
    bad_image["Config"]["Labels"] = {
        B.PROBE_ROLE_LABEL: "substituted-role"
    }
    image_runner = FakeProbeDockerRunner(
        spec,
        launched,
        probe_request_2,
        parent_record=parent,
        image_records=[bad_image],
    )
    with pytest.raises(
        B.ContainerContractError, match="token-free role"
    ):
        B.DockerWorkspaceProbeExecutor(
            B.DockerContainerBackend(image_runner),
            image_reference=PROBE_IMAGE_REFERENCE,
        ).run_probe(
            spec=spec, launched=launched, request=probe_request_2
        )
    assert image_runner.created is False


def test_workspace_probe_recipe_contains_no_campaign_modules():
    recipe = (
        Path(__file__).parent
        / "container"
        / "Containerfile.arc-agi3-workspace-probe"
    ).read_text(encoding="utf-8")
    assert "@sha256:" in recipe
    assert "USER 65534:65534" in recipe
    assert 'ENTRYPOINT ["/usr/local/bin/python3", "-I"]' in recipe
    assert "COPY " not in recipe
    assert "arc_agi3_" not in recipe
    assert "/arc/input" not in recipe
    assert "/arc/export" not in recipe
    assert "/run/arc-agi3" not in recipe
