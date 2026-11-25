from __future__ import annotations

import hashlib
import io
import socket
import subprocess
import sys

import pytest

import arc_agi3_arena_volume_transport as T


def test_read_exact_handles_fragmented_binary_stream():
    assert T._read_exact(io.BytesIO(b"abcdef"), 4) == b"abcd"
    assert T._read_exact(io.BytesIO(b"ab"), 4) == b"ab"


def test_echo_probe_rejects_nonabsolute_docker_and_ids():
    try:
        T.run_echo_probe(
            docker="docker",
            relay_container_id="1" * 64,
            client_container_id="2" * 64,
            payload=b"nonce",
        )
    except T.ArenaVolumeTransportError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("relative Docker authority was accepted")


def test_attached_relay_is_full_duplex_and_emits_exact_receipt():
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            (
                "import os\n"
                "while True:\n"
                " b=os.read(0,65536)\n"
                " if not b: break\n"
                " os.write(1,b)\n"
            ),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client, arena = socket.socketpair()
    handle = T.AttachedArenaRelay.start(
        process=process,
        arena_socket=arena,
        relay_argv=("/usr/bin/docker", "container", "attach", "1" * 64),
        relay_container_id="1" * 64,
        arena_socket_identity_sha256="2" * 64,
    )
    try:
        client.settimeout(5)
        client.sendall(b"full-duplex")
        assert client.recv(len(b"full-duplex")) == b"full-duplex"
        client.shutdown(socket.SHUT_WR)
        receipt = handle.finish(timeout_seconds=5)
    finally:
        client.close()
        if not handle._finished:
            handle.abort()
    expected = hashlib.sha256(b"full-duplex").hexdigest()
    assert receipt["status"] == "PASS"
    assert receipt["arena_to_container_bytes"] == len(b"full-duplex")
    assert receipt["container_to_arena_bytes"] == len(b"full-duplex")
    assert receipt["arena_to_container_sha256"] == expected
    assert receipt["container_to_arena_sha256"] == expected
    assert receipt["relay_exit_code"] == 0
    assert receipt["threads_stopped"] is True


def test_attached_relay_rejects_stderr_even_with_zero_exit():
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import os;os.write(2,b'unexpected')",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client, arena = socket.socketpair()
    handle = T.AttachedArenaRelay.start(
        process=process,
        arena_socket=arena,
        relay_argv=("/usr/bin/docker", "container", "attach", "3" * 64),
        relay_container_id="3" * 64,
        arena_socket_identity_sha256="4" * 64,
    )
    try:
        client.shutdown(socket.SHUT_WR)
        with pytest.raises(
            T.ArenaVolumeTransportError,
            match="terminate cleanly",
        ):
            handle.finish(timeout_seconds=5)
    finally:
        client.close()
        if not handle._finished:
            handle.abort()
