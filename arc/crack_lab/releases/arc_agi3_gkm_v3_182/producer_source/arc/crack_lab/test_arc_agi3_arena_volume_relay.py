from __future__ import annotations

import os
import socket
import threading
import uuid
from types import SimpleNamespace

import pytest

import arc_agi3_arena_volume_relay as R


def test_relay_rejects_wrong_identity_and_writable_socket_root(
    tmp_path, monkeypatch
):
    root = tmp_path / "arena"
    root.mkdir(mode=0o777)
    monkeypatch.setattr(R, "SOCKET_PATH", root / "arena.sock")
    monkeypatch.setattr(
        R, "READINESS_PATH", tmp_path / "run" / "readiness.json"
    )
    monkeypatch.setattr(R.os, "getuid", lambda: 0)
    args = SimpleNamespace(
        socket_path=str(R.SOCKET_PATH),
        campaign_id=str(uuid.uuid4()),
        generation_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        readiness_nonce="a" * 64,
    )
    with pytest.raises(
        R.ArenaVolumeRelayError, match="socket root"
    ):
        R.run(args)


def test_relay_client_to_host_stream_is_byte_exact(monkeypatch):
    client, relay = socket.socketpair()
    observed = bytearray()
    monkeypatch.setattr(
        R.os,
        "write",
        lambda descriptor, payload: (
            observed.extend(bytes(payload))
            or len(payload)
        )
        if descriptor == 1
        else os.write(descriptor, payload),
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=R._relay_client_to_stdout,
        args=(relay, stop),
    )
    thread.start()
    try:
        client.sendall(b"client-to-host")
        client.shutdown(socket.SHUT_WR)
        thread.join(timeout=5)
    finally:
        client.close()
        relay.close()
    assert not thread.is_alive()
    assert observed == b"client-to-host"
    assert stop.is_set()
