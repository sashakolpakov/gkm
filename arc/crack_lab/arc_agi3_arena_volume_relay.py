#!/usr/bin/env python3
"""Networkless stdio-to-Unix relay for the Colima Arena transport.

The trusted host attaches to this container over Docker stdio.  The solver
connects to the Unix socket through a per-attempt named volume.  Consequently
neither container needs a network namespace with routes, and no macOS-host
Unix socket is bind-mounted through Colima's VM boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import threading
import uuid
from pathlib import Path
from typing import Sequence


SOCKET_PATH = Path("/arena/arena.sock")
READINESS_PATH = Path("/run/arc-agi3-arena-relay/readiness.json")
MAX_RELAY_BLOCK = 64 * 1024


class ArenaVolumeRelayError(RuntimeError):
    """The relay could not establish its exact isolated transport."""


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


def _canonical_uuid(value: str) -> str:
    parsed = uuid.UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ArenaVolumeRelayError("relay identity is not a UUIDv4")
    return value


def _write_new(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_json(body)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ArenaVolumeRelayError(
                    "relay readiness write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relay_client_to_stdout(
    client: socket.socket, stop: threading.Event
) -> None:
    try:
        while not stop.is_set():
            block = client.recv(MAX_RELAY_BLOCK)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(1, view)
                if written <= 0:
                    raise ArenaVolumeRelayError(
                        "relay stdout write made no progress"
                    )
                view = view[written:]
    except (OSError, ArenaVolumeRelayError):
        pass
    finally:
        stop.set()


def run(args: argparse.Namespace) -> int:
    if (
        os.getuid() != 0
        or Path(args.socket_path) != SOCKET_PATH
        or len(args.readiness_nonce) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.readiness_nonce
        )
    ):
        raise ArenaVolumeRelayError("relay launch binding differs")
    campaign_id = _canonical_uuid(args.campaign_id)
    generation_id = _canonical_uuid(args.generation_id)
    attempt_id = _canonical_uuid(args.attempt_id)
    parent = SOCKET_PATH.parent
    metadata = parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or SOCKET_PATH.exists()
        or SOCKET_PATH.is_symlink()
    ):
        raise ArenaVolumeRelayError(
            "relay named-volume socket root differs"
        )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o666)
    listener.listen(1)
    listener.settimeout(1)
    socket_metadata = SOCKET_PATH.lstat()
    readiness = {
        "schema": 1,
        "kind": "arc_agi3_arena_volume_relay_readiness",
        "status": "READY",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "attempt_id": attempt_id,
        "readiness_nonce": args.readiness_nonce,
        "relay_pid": os.getpid(),
        "socket_path": str(SOCKET_PATH),
        "socket_mode": stat.S_IMODE(socket_metadata.st_mode),
        "network_mode_required": "none",
        "transport": "docker-attach-stdio+named-volume-unix",
    }
    _write_new(READINESS_PATH, readiness)
    stop = threading.Event()
    for selected in (signal.SIGTERM, signal.SIGINT):
        signal.signal(selected, lambda *_unused: stop.set())
    client: socket.socket | None = None
    try:
        while not stop.is_set():
            try:
                client, _ = listener.accept()
                break
            except socket.timeout:
                continue
        if client is None:
            return 0
        listener.close()
        reader = threading.Thread(
            target=_relay_client_to_stdout,
            args=(client, stop),
            daemon=False,
        )
        reader.start()
        while not stop.is_set():
            block = os.read(0, MAX_RELAY_BLOCK)
            if not block:
                break
            client.sendall(block)
        stop.set()
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        reader.join(timeout=5)
        if reader.is_alive():
            raise ArenaVolumeRelayError(
                "relay reader did not terminate"
            )
        return 0
    finally:
        stop.set()
        listener.close()
        if client is not None:
            client.close()
        try:
            SOCKET_PATH.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--readiness-nonce", required=True)
    parser.add_argument("--print-source-sha256", action="store_true")
    args = parser.parse_args(argv)
    if args.print_source_sha256:
        print(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return 0
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArenaVolumeRelayError, OSError, ValueError) as error:
        print(f"Arena volume relay failed: {error}", file=sys.stderr)
        raise SystemExit(70)
