#!/usr/bin/env python3
"""Host half of the networkless Colima Arena named-volume transport."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence


CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
MAX_RELAY_BLOCK = 64 * 1024
MAX_RELAY_STDERR_BYTES = 64 * 1024


class ArenaVolumeTransportError(RuntimeError):
    """The Docker-attach/named-volume transport failed closed."""


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            break
        result.extend(block)
    return bytes(result)


def run_echo_probe(
    *,
    docker: str,
    relay_container_id: str,
    client_container_id: str,
    payload: bytes,
) -> dict[str, object]:
    if (
        not Path(docker).is_absolute()
        or not os.access(docker, os.X_OK)
        or not CONTAINER_ID_RE.fullmatch(relay_container_id)
        or not CONTAINER_ID_RE.fullmatch(client_container_id)
        or not payload
        or len(payload) > 4096
    ):
        raise ArenaVolumeTransportError(
            "Arena volume probe binding is malformed"
        )
    environment = {
        "DOCKER_CONFIG": os.environ.get(
            "DOCKER_CONFIG", str(Path.home() / ".docker")
        ),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    }
    if not environment["DOCKER_HOST"]:
        environment.pop("DOCKER_HOST")
    relay_argv = (
        docker,
        "container",
        "attach",
        relay_container_id,
    )
    client_argv = (
        docker,
        "container",
        "start",
        "--attach",
        client_container_id,
    )
    relay = subprocess.Popen(
        relay_argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd="/",
        close_fds=True,
    )
    client: subprocess.Popen[bytes] | None = None
    try:
        client = subprocess.Popen(
            client_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd="/",
            close_fds=True,
        )
        assert relay.stdin is not None
        assert relay.stdout is not None
        assert client.stdout is not None
        observed = _read_exact(relay.stdout, len(payload))
        if observed != payload:
            raise ArenaVolumeTransportError(
                "client-to-host named-volume relay bytes differ"
            )
        relay.stdin.write(observed)
        relay.stdin.flush()
        expected_client = payload.hex().encode("ascii")
        client_output = _read_exact(
            client.stdout, len(expected_client)
        )
        client_returncode = client.wait(timeout=20)
        relay.stdin.close()
        relay_returncode = relay.wait(timeout=20)
        client_stderr = (
            client.stderr.read()
            if client.stderr is not None
            else b""
        )
        relay_stderr = (
            relay.stderr.read()
            if relay.stderr is not None
            else b""
        )
        if (
            client_output != expected_client
            or client_returncode != 0
            or relay_returncode != 0
            or client_stderr
            or relay_stderr
        ):
            raise ArenaVolumeTransportError(
                "host-to-client named-volume relay proof failed"
            )
        return {
            "schema": 1,
            "kind": "arc_agi3_arena_volume_transport_probe",
            "status": "PASS",
            "transport":
                "docker-attach-stdio+named-volume-unix",
            "relay_container_id": relay_container_id,
            "client_container_id": client_container_id,
            "relay_argv_sha256": hashlib.sha256(
                canonical_json(list(relay_argv))
            ).hexdigest(),
            "client_argv_sha256": hashlib.sha256(
                canonical_json(list(client_argv))
            ).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
            "client_to_host_byte_exact": True,
            "host_to_client_byte_exact": True,
            "relay_exit_code": relay_returncode,
            "client_exit_code": client_returncode,
        }
    finally:
        for process in (client, relay):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@dataclass
class _DirectionDigest:
    byte_count: int = 0
    digest: object = None

    def __post_init__(self) -> None:
        self.digest = hashlib.sha256()

    def update(self, block: bytes) -> None:
        self.byte_count += len(block)
        self.digest.update(block)

    def snapshot(self) -> tuple[int, str]:
        return self.byte_count, self.digest.hexdigest()


def _write_all(destination: BinaryIO, block: bytes) -> None:
    view = memoryview(block)
    while view:
        written = destination.write(view)
        if written is None:
            written = len(view)
        if written <= 0:
            raise ArenaVolumeTransportError(
                "Arena relay stream write made no progress"
            )
        view = view[written:]


def _read_block(source: BinaryIO) -> bytes:
    try:
        return os.read(source.fileno(), MAX_RELAY_BLOCK)
    except (AttributeError, io.UnsupportedOperation):
        read1 = getattr(source, "read1", None)
        if callable(read1):
            return read1(MAX_RELAY_BLOCK)
        return source.read(MAX_RELAY_BLOCK)


def _socket_to_stream(
    source: socket.socket,
    destination: BinaryIO,
    abort: threading.Event,
    evidence: _DirectionDigest,
    errors: list[str],
) -> None:
    try:
        while not abort.is_set():
            block = source.recv(MAX_RELAY_BLOCK)
            if not block:
                break
            _write_all(destination, block)
            destination.flush()
            evidence.update(block)
    except BaseException as error:
        errors.append(type(error).__name__)
    finally:
        try:
            destination.close()
        except OSError:
            pass


def _stream_to_socket(
    source: BinaryIO,
    destination: socket.socket,
    abort: threading.Event,
    evidence: _DirectionDigest,
    errors: list[str],
) -> None:
    try:
        while not abort.is_set():
            block = _read_block(source)
            if not block:
                break
            destination.sendall(block)
            evidence.update(block)
    except BaseException as error:
        errors.append(type(error).__name__)
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _drain_stderr(
    source: BinaryIO,
    abort: threading.Event,
    output: bytearray,
    errors: list[str],
) -> None:
    try:
        while not abort.is_set():
            block = _read_block(source)
            if not block:
                break
            output.extend(block)
            if len(output) > MAX_RELAY_STDERR_BYTES:
                errors.append("RelayStderrOverflow")
                abort.set()
                break
    except BaseException as error:
        errors.append(type(error).__name__)


@dataclass
class AttachedArenaRelay:
    """Live host-owned relay between Docker attach and the Arena server."""

    process: subprocess.Popen[bytes]
    arena_socket: socket.socket
    abort_event: threading.Event
    arena_to_container: threading.Thread
    container_to_arena: threading.Thread
    stderr_thread: threading.Thread
    arena_to_container_digest: _DirectionDigest
    container_to_arena_digest: _DirectionDigest
    stderr: bytearray
    errors: list[str]
    relay_argv: tuple[str, ...]
    relay_container_id: str
    arena_socket_identity_sha256: str
    _finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        process: subprocess.Popen[bytes],
        arena_socket: socket.socket,
        relay_argv: tuple[str, ...],
        relay_container_id: str,
        arena_socket_identity_sha256: str,
    ) -> "AttachedArenaRelay":
        if (
            process.stdin is None
            or process.stdout is None
            or process.stderr is None
            or not CONTAINER_ID_RE.fullmatch(relay_container_id)
        ):
            raise ArenaVolumeTransportError(
                "attached Arena relay process lacks exact stdio"
            )
        abort = threading.Event()
        upstream = _DirectionDigest()
        downstream = _DirectionDigest()
        stderr = bytearray()
        errors: list[str] = []
        arena_to_container = threading.Thread(
            target=_socket_to_stream,
            args=(
                arena_socket,
                process.stdin,
                abort,
                upstream,
                errors,
            ),
            name="arena-host-to-volume-relay",
            daemon=False,
        )
        container_to_arena = threading.Thread(
            target=_stream_to_socket,
            args=(
                process.stdout,
                arena_socket,
                abort,
                downstream,
                errors,
            ),
            name="arena-volume-relay-to-host",
            daemon=False,
        )
        stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(process.stderr, abort, stderr, errors),
            name="arena-volume-relay-stderr",
            daemon=False,
        )
        value = cls(
            process=process,
            arena_socket=arena_socket,
            abort_event=abort,
            arena_to_container=arena_to_container,
            container_to_arena=container_to_arena,
            stderr_thread=stderr_thread,
            arena_to_container_digest=upstream,
            container_to_arena_digest=downstream,
            stderr=stderr,
            errors=errors,
            relay_argv=relay_argv,
            relay_container_id=relay_container_id,
            arena_socket_identity_sha256=(
                arena_socket_identity_sha256
            ),
        )
        for thread in (
            arena_to_container,
            container_to_arena,
            stderr_thread,
        ):
            thread.start()
        return value

    def finish(self, *, timeout_seconds: float = 30.0) -> dict[str, object]:
        if self._finished:
            raise ArenaVolumeTransportError(
                "attached Arena relay finish is not repeatable"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ArenaVolumeTransportError(
                "attached Arena relay timeout is invalid"
            )
        for thread in (
            self.arena_to_container,
            self.container_to_arena,
        ):
            thread.join(timeout=timeout_seconds)
        try:
            returncode = self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            self.abort()
            raise ArenaVolumeTransportError(
                "attached Arena relay exceeded its terminal bound"
            ) from error
        self.stderr_thread.join(timeout=timeout_seconds)
        self._finished = True
        try:
            self.arena_socket.close()
        except OSError:
            pass
        threads_stopped = all(
            not thread.is_alive()
            for thread in (
                self.arena_to_container,
                self.container_to_arena,
                self.stderr_thread,
            )
        )
        if (
            not threads_stopped
            or returncode != 0
            or self.errors
            or self.stderr
        ):
            self.abort()
            raise ArenaVolumeTransportError(
                "attached Arena relay did not terminate cleanly"
            )
        upstream_bytes, upstream_sha256 = (
            self.arena_to_container_digest.snapshot()
        )
        downstream_bytes, downstream_sha256 = (
            self.container_to_arena_digest.snapshot()
        )
        return {
            "schema": 1,
            "kind": "arc_agi3_attached_arena_relay",
            "status": "PASS",
            "transport":
                "docker-attach-stdio+named-volume-unix",
            "relay_container_id": self.relay_container_id,
            "relay_argv_sha256": hashlib.sha256(
                canonical_json(list(self.relay_argv))
            ).hexdigest(),
            "arena_socket_identity_sha256":
                self.arena_socket_identity_sha256,
            "arena_to_container_bytes": upstream_bytes,
            "arena_to_container_sha256": upstream_sha256,
            "container_to_arena_bytes": downstream_bytes,
            "container_to_arena_sha256": downstream_sha256,
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "relay_exit_code": returncode,
            "threads_stopped": True,
        }

    def close(self) -> None:
        self.abort()

    def abort(self) -> None:
        self.abort_event.set()
        try:
            self.arena_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.arena_socket.close()
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        for thread in (
            self.arena_to_container,
            self.container_to_arena,
            self.stderr_thread,
        ):
            thread.join(timeout=5)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._finished = True


def _endpoint_identity_sha256(path: Path) -> str:
    if (
        not path.is_absolute()
        or path.is_symlink()
    ):
        raise ArenaVolumeTransportError(
            "Arena host socket path is not canonical"
        )
    metadata = path.lstat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ArenaVolumeTransportError(
            "Arena host socket is not private and unaliased"
        )
    return hashlib.sha256(canonical_json({
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
    })).hexdigest()


def start_attached_relay(
    *,
    docker: str,
    docker_socket: Path,
    docker_config: Path,
    relay_container_id: str,
    arena_socket_path: Path,
    connect_timeout_seconds: float = 30.0,
) -> AttachedArenaRelay:
    """Attach one exact relay container to one private host Arena socket."""

    docker_path = Path(docker)
    if (
        not docker_path.is_absolute()
        or not os.access(docker_path, os.X_OK)
        or not docker_socket.is_absolute()
        or docker_socket.is_symlink()
        or not stat.S_ISSOCK(docker_socket.lstat().st_mode)
        or not docker_config.is_absolute()
        or docker_config.is_symlink()
        or not docker_config.is_dir()
        or not CONTAINER_ID_RE.fullmatch(relay_container_id)
        or isinstance(connect_timeout_seconds, bool)
        or not isinstance(connect_timeout_seconds, (int, float))
        or connect_timeout_seconds <= 0
    ):
        raise ArenaVolumeTransportError(
            "attached Arena relay launch binding is malformed"
        )
    socket_identity = _endpoint_identity_sha256(
        arena_socket_path
    )
    argv = (
        str(docker_path),
        "container",
        "attach",
        relay_container_id,
    )
    environment = {
        "DOCKER_CONFIG": str(docker_config),
        "DOCKER_HOST": f"unix://{docker_socket}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    }
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd="/",
        close_fds=True,
    )
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(float(connect_timeout_seconds))
        connection.connect(str(arena_socket_path))
        connection.settimeout(None)
        return AttachedArenaRelay.start(
            process=process,
            arena_socket=connection,
            relay_argv=argv,
            relay_container_id=relay_container_id,
            arena_socket_identity_sha256=socket_identity,
        )
    except BaseException:
        connection.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--docker", required=True)
    parser.add_argument("--relay-container-id", required=True)
    parser.add_argument("--client-container-id", required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    result = run_echo_probe(
        docker=args.docker,
        relay_container_id=args.relay_container_id,
        client_container_id=args.client_container_id,
        payload=args.payload.encode("ascii"),
    )
    sys.stdout.buffer.write(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArenaVolumeTransportError, OSError) as error:
        print(f"Arena volume transport failed: {error}", file=sys.stderr)
        raise SystemExit(70)
