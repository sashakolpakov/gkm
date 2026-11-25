from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import socket
import struct
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

import arc_agi3_codex_app_server_transport as Transport
import arc_agi3_proposer_worker as Worker


ATTEMPT_ID = "attempt-bridge-replay"
TOKEN = "a" * 64
SESSION_NONCE = "b" * 32
CHALLENGE_NONCE = "c" * 32


def _server() -> Worker.ProposerBridgeServer:
    server = Worker.ProposerBridgeServer.__new__(
        Worker.ProposerBridgeServer
    )
    server.policy = Worker.BridgePolicy(
        campaign_id="campaign",
        generation_id="generation",
        attempt_id=ATTEMPT_ID,
        game="game",
        target_level=1,
        frontier_sha256="d" * 64,
        parent_checkpoint_sha256="e" * 64,
        operation_allowlist=("candidate_publish",),
        exec_allowlist=(),
        max_request_bytes=1024 * 1024,
        max_response_bytes=1024 * 1024,
        max_file_bytes=1024 * 1024,
        max_total_export_bytes=1024 * 1024,
        max_processes=1,
        max_exec_seconds=1,
    )
    server.token = TOKEN
    server.connection_challenge = CHALLENGE_NONCE
    server.session_nonce = SESSION_NONCE
    server.sequence = 0
    server.mutation_sequence = 0
    server.cached = {}
    server.executions = 0

    def execute(self, operation, arguments):
        assert operation == "candidate_publish"
        self.executions += 1
        return {
            "published": True,
            "arguments_sha256": hashlib.sha256(
                Worker._canonical_json(arguments)
            ).hexdigest(),
        }

    server._execute = types.MethodType(execute, server)
    return server


def _request(
    *,
    sequence: int,
    mutation_sequence: int,
    request_id: str,
) -> dict:
    body = {
        "schema": Worker.SCHEMA,
        "kind": "arc_agi3_contiguous_bridge_request",
        "protocol_version": Worker.PROTOCOL_VERSION,
        "attempt_id": ATTEMPT_ID,
        "request_id": request_id,
        "sequence": sequence,
        "session_nonce": SESSION_NONCE,
        "operation": "candidate_publish",
        "mutation_id": f"{ATTEMPT_ID}:{mutation_sequence:08d}",
        "challenge_nonce": CHALLENGE_NONCE,
        "arguments": {"candidate_path": [1], "exports": {}},
    }
    return {
        **body,
        "auth_hmac": hmac.new(
            TOKEN.encode("ascii"),
            Worker._canonical_json(body),
            hashlib.sha256,
        ).hexdigest(),
    }


def test_cache_capacity_is_reserved_before_mutating_publish():
    server = _server()
    for index in range(Worker.MAX_CACHED_RESPONSES - 1):
        server.cached[f"prior-{index}"] = (
            "0" * 64,
            {"prior": index},
        )
    server.sequence = Worker.MAX_CACHED_RESPONSES - 1
    server.mutation_sequence = Worker.MAX_CACHED_RESPONSES - 1
    admitted_id = str(uuid.uuid4())
    admitted = _request(
        sequence=Worker.MAX_CACHED_RESPONSES,
        mutation_sequence=Worker.MAX_CACHED_RESPONSES,
        request_id=admitted_id,
    )
    response = server.dispatch(admitted)
    assert response["success"] is True
    assert server.executions == 1
    assert server.dispatch(admitted) == response
    assert server.executions == 1

    rejected = _request(
        sequence=Worker.MAX_CACHED_RESPONSES + 1,
        mutation_sequence=Worker.MAX_CACHED_RESPONSES + 1,
        request_id=str(uuid.uuid4()),
    )
    with pytest.raises(
        Worker.ProposerBridgeError,
        match="idempotency cache exhausted",
    ):
        server.dispatch(rejected)
    assert server.executions == 1
    assert server.sequence == Worker.MAX_CACHED_RESPONSES
    assert server.mutation_sequence == Worker.MAX_CACHED_RESPONSES


class _DroppedFirstResponseSocket:
    def __init__(self, server: Worker.ProposerBridgeServer):
        self.server = server
        self.sent: list[bytes] = []
        self.responses: list[bytes] = []
        self.drop_first = True

    def sendall(self, wire: bytes) -> None:
        self.sent.append(wire)
        request = json.loads(wire)
        response = self.server.dispatch(request)
        if self.drop_first:
            self.drop_first = False
        else:
            self.responses.append(
                Transport.canonical_json(response) + b"\n"
            )

    def recv(self, _maximum: int) -> bytes:
        if not self.responses:
            raise socket.timeout("injected lost response")
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def test_bridge_client_replays_exact_mutation_bytes_after_lost_response():
    server = _server()
    fake_socket = _DroppedFirstResponseSocket(server)
    client = Transport.BridgeClient.__new__(Transport.BridgeClient)
    client.socket_path = None
    client.attempt_id = ATTEMPT_ID
    client._token = TOKEN
    client._callback = None
    client._socket = fake_socket
    client._recv_buffer = bytearray()
    client._completed_responses = {}
    client.challenge_nonce = CHALLENGE_NONCE
    client.sequence = 0
    client.mutation_sequence = 0
    client.session_nonce = SESSION_NONCE
    client.handshake_request_sha256 = None
    client.handshake_response_sha256 = None
    client.handshake_result = None

    result = client.call(
        "candidate_publish",
        {"candidate_path": [1], "exports": {}},
        idempotency_key="publish-once",
    )
    assert result["published"] is True
    assert server.executions == 1
    assert len(fake_socket.sent) == 2
    assert fake_socket.sent[0] == fake_socket.sent[1]
    assert client.sequence == 1
    assert client.mutation_sequence == 1


class _ImmediateResponseSocket:
    def __init__(self, server: Worker.ProposerBridgeServer):
        self.server = server
        self.responses: list[bytes] = []

    def sendall(self, wire: bytes) -> None:
        response = self.server.dispatch(json.loads(wire))
        self.responses.append(Transport.canonical_json(response) + b"\n")

    def recv(self, _maximum: int) -> bytes:
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def _bare_client(fake_socket) -> Transport.BridgeClient:
    client = Transport.BridgeClient.__new__(Transport.BridgeClient)
    client.socket_path = None
    client.attempt_id = ATTEMPT_ID
    client._token = TOKEN
    client._callback = None
    client._socket = fake_socket
    client._recv_buffer = bytearray()
    client._completed_responses = {}
    client.challenge_nonce = CHALLENGE_NONCE
    client.sequence = 0
    client.mutation_sequence = 0
    client.session_nonce = SESSION_NONCE
    client.handshake_request_sha256 = None
    client.handshake_response_sha256 = None
    client.handshake_result = None
    return client


def test_error_response_advances_client_and_server_sequences():
    server = _server()
    original_execute = server._execute

    def execute(self, operation, arguments):
        if arguments["candidate_path"] == [7]:
            raise Worker.ProposerBridgeError("injected application rejection")
        return original_execute(operation, arguments)

    server._execute = types.MethodType(execute, server)
    client = _bare_client(_ImmediateResponseSocket(server))
    with pytest.raises(
        Transport.AppServerTransportError,
        match="injected application rejection",
    ):
        client.call(
            "candidate_publish",
            {"candidate_path": [7], "exports": {}},
            idempotency_key="rejected",
        )
    assert (client.sequence, client.mutation_sequence) == (1, 1)
    assert (server.sequence, server.mutation_sequence) == (1, 1)

    result = client.call(
        "candidate_publish",
        {"candidate_path": [1], "exports": {}},
        idempotency_key="accepted-after-rejection",
    )
    assert result["published"] is True
    assert (client.sequence, client.mutation_sequence) == (2, 2)
    assert (server.sequence, server.mutation_sequence) == (2, 2)


class _TargetRoot:
    def __init__(self) -> None:
        self.levels_completed = 0
        self.actions = (1, 2)
        self.observe_calls = 0

    def step(self, action, *coordinates):
        assert action == 1
        assert coordinates == ()
        self.levels_completed = 1
        return [[99]]

    def observe(self):
        self.observe_calls += 1
        return [[99]]

    def terminal(self):
        return False


def _target_server(tmp_path: Path) -> tuple[
    Worker.ProposerBridgeServer, _TargetRoot
]:
    workspace = tmp_path / "workspace"
    export = tmp_path / "export"
    workspace.mkdir()
    export.mkdir()
    for name in ("legs.py", "players.py", "solve.py"):
        (workspace / name).write_text(
            f"# exact pre-debrief {name}\n", encoding="utf-8"
        )
    server = Worker.ProposerBridgeServer.__new__(
        Worker.ProposerBridgeServer
    )
    server.policy = Worker.BridgePolicy(
        campaign_id="campaign",
        generation_id="generation",
        attempt_id=ATTEMPT_ID,
        game="game",
        target_level=1,
        frontier_sha256="d" * 64,
        parent_checkpoint_sha256="e" * 64,
        operation_allowlist=Transport.BRIDGE_OPERATION_ALLOWLIST,
        exec_allowlist=(),
        max_request_bytes=1024 * 1024,
        max_response_bytes=1024 * 1024,
        max_file_bytes=1024 * 1024,
        max_total_export_bytes=1024 * 1024,
        max_processes=1,
        max_exec_seconds=1,
    )
    server.workspace = workspace
    server.export = export
    server.workspace_fd = Worker._open_root(
        workspace, label="test workspace"
    )
    server.export_fd = Worker._open_root(export, label="test export")
    root = _TargetRoot()
    server.arena_client = types.SimpleNamespace(
        root=root,
        binding_sha256="f" * 64,
    )
    server.token = TOKEN
    server.connection_challenge = CHALLENGE_NONCE
    server.session_nonce = SESSION_NONCE
    server.cached = {}
    server.sequence = 3
    server.mutation_sequence = 2
    server.exploration_suffix = []
    server.target_boundary = None
    server.target_boundary_sha256 = None
    server.boundary_workspace_files = None
    server._active_request = {
        "operation": "arena_step",
        "request_id": str(uuid.uuid4()),
        "sequence": 4,
        "mutation_id": f"{ATTEMPT_ID}:00000003",
        "arguments": {"action": 1},
    }
    server.published = False
    return server, root


def test_target_step_freezes_workspace_without_next_level_observation(
    tmp_path: Path,
) -> None:
    server, root = _target_server(tmp_path)
    try:
        result = server._execute("arena_step", {"action": 1})
        assert set(result) == {
            "target_reached",
            "boundary",
            "boundary_sha256",
        }
        assert result["target_reached"] is True
        assert root.observe_calls == 0
        assert "frame" not in json.dumps(result)
        assert "actions" not in json.dumps(result)
        assert result["boundary"]["bridge_sequence"] == 4
        assert (
            result["boundary"]["bridge_mutation_id"]
            == f"{ATTEMPT_ID}:00000003"
        )
        assert result["boundary"]["workspace_file_count"] == 3
        with pytest.raises(
            Worker.ProposerBridgeError,
            match="target boundary is frozen",
        ):
            server._execute("arena_observe", {})
        with pytest.raises(
            Worker.ProposerBridgeError,
            match="target boundary is frozen",
        ):
            server._execute(
                "workspace_write",
                {"path": "solve.py", "text": "# post-target\n"},
            )
    finally:
        Worker.ProposerBridgeServer.close(server)


def test_candidate_source_must_equal_frozen_pre_debrief_bytes(
    tmp_path: Path,
) -> None:
    server, _root = _target_server(tmp_path)
    try:
        server._execute("arena_step", {"action": 1})
        (server.workspace / "legs.py").write_text(
            "# illicit post-target mutation\n", encoding="utf-8"
        )
        with pytest.raises(
            Worker.ProposerBridgeError,
            match="pre-debrief workspace boundary",
        ):
            server._publish(
                {
                    "candidate_path": [1],
                    "exports": {
                        "legs.py": "legs.py",
                        "players.py": "players.py",
                        "solve.py": "solve.py",
                    },
                },
                candidate=True,
            )
    finally:
        Worker.ProposerBridgeServer.close(server)


def test_wip_publish_separates_broad_context_from_solver_source(
    tmp_path: Path,
) -> None:
    server, _root = _target_server(tmp_path)
    try:
        (server.workspace / "notes.txt").write_text(
            "retained exploration context\n", encoding="utf-8"
        )
        (server.export / Worker.WORKER_OUTCOME_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
        result = server._publish(
            {
                "exports": {
                    "legs.py": "wip/solver_source/legs.py",
                    "players.py": "wip/solver_source/players.py",
                    "solve.py": "wip/solver_source/solve.py",
                    "notes.txt": "wip/context/notes.txt",
                },
            },
            candidate=False,
        )
        assert result["outcome"] == "wip"
        manifest = json.loads(
            (server.export / Worker.WIP_MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
        )
        assert manifest["wip_root_relative_path"] == "wip"
        assert (
            manifest["solver_source_relative_path"]
            == "wip/solver_source"
        )
        broad = {
            path.relative_to(server.export / "wip").as_posix():
                path.read_bytes()
            for path in (server.export / "wip").rglob("*")
            if path.is_file()
        }
        source = {
            path.name: path.read_bytes()
            for path in (
                server.export / "wip" / "solver_source"
            ).iterdir()
        }
        assert manifest["wip_tree_sha256"] == (
            Worker._payload_tree_sha256(broad)
        )
        assert manifest["solver_source_tree_sha256"] == (
            Worker._payload_tree_sha256(source)
        )
    finally:
        Worker.ProposerBridgeServer.close(server)


def test_wip_publish_rejects_legacy_flat_or_incomplete_source(
    tmp_path: Path,
) -> None:
    server, _root = _target_server(tmp_path)
    try:
        (server.export / Worker.WORKER_OUTCOME_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
        with pytest.raises(
            Worker.ProposerBridgeError,
            match="solver_source files",
        ):
            server._publish(
                {"exports": {"legs.py": "wip/legs.py"}},
                candidate=False,
            )
        with pytest.raises(
            Worker.ProposerBridgeError,
            match="closed source schema",
        ):
            server._publish(
                {
                    "exports": {
                        "legs.py": "wip/solver_source/legs.py",
                        "players.py":
                            "wip/solver_source/players.py",
                    },
                },
                candidate=False,
            )
    finally:
        Worker.ProposerBridgeServer.close(server)


def test_target_boundary_validator_rejects_any_next_level_payload(
    tmp_path: Path,
) -> None:
    server, _root = _target_server(tmp_path)
    try:
        result = server._execute("arena_step", {"action": 1})
        request = dict(server._active_request)
        assert Transport._validate_target_boundary_result(
            result,
            attempt_id=ATTEMPT_ID,
            request=request,
            target_level=1,
        ) == result["boundary_sha256"]
        tainted = json.loads(json.dumps(result))
        tainted["frame"] = [[99]]
        with pytest.raises(
            Transport.AppServerTransportError,
            match="malformed",
        ):
            Transport._validate_target_boundary_result(
                tainted,
                attempt_id=ATTEMPT_ID,
                request=request,
                target_level=1,
            )
    finally:
        Worker.ProposerBridgeServer.close(server)


def test_bridge_client_runs_boundary_callback_before_return_and_closes_lane(
    tmp_path: Path,
) -> None:
    server, _root = _target_server(tmp_path)
    server.sequence = 0
    server.mutation_sequence = 0
    observed: list[tuple[dict, dict]] = []
    client = _bare_client(_ImmediateResponseSocket(server))
    client.handshake_result = {"target_level": 1}
    client.target_boundary_sha256 = None
    client._target_boundary_callback = (
        lambda request, response: observed.append(
            (dict(request), dict(response))
        )
    )
    try:
        result = client.call(
            "arena_step",
            {"action": 1},
            idempotency_key="exact-winning-step",
        )
        assert result["target_reached"] is True
        assert len(observed) == 1
        assert observed[0][0]["mutation_id"] == (
            f"{ATTEMPT_ID}:00000001"
        )
        assert client.target_boundary_sha256 == (
            result["boundary_sha256"]
        )
        sequence = server.sequence
        with pytest.raises(
            Transport.AppServerTransportError,
            match="target boundary is frozen",
        ):
            client.call(
                "arena_observe",
                {},
                idempotency_key="forbidden-post-target-observe",
            )
        assert server.sequence == sequence
    finally:
        Worker.ProposerBridgeServer.close(server)


def _socket_record(connection: socket.socket) -> dict:
    wire = bytearray()
    while b"\n" not in wire:
        block = connection.recv(65536)
        if not block:
            raise AssertionError("socket closed before one record")
        wire.extend(block)
    line, _, remainder = wire.partition(b"\n")
    assert not remainder
    return json.loads(line)


def _connect_when_listening(path: Path) -> socket.socket:
    deadline = time.monotonic() + 2
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(2)
        try:
            connection.connect(str(path))
            return connection
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            last_error = exc
            connection.close()
            time.sleep(0.001)
    raise AssertionError(
        f"bridge did not become connectable: {last_error}"
    )


def test_queued_lost_response_reconnect_is_stable_under_race(
    request, monkeypatch
):
    monkeypatch.setattr(Worker, "RECONNECT_GRACE_SECONDS", 0.01)
    socket_root = Path(tempfile.mkdtemp(
        prefix=".a3s_",
        dir=tempfile.gettempdir(),
    ))
    request.addfinalizer(
        lambda: shutil.rmtree(socket_root, ignore_errors=True)
    )
    for iteration in range(25):
        server = _server()
        # AF_UNIX paths are short on macOS; pytest's nested temporary path can
        # exceed that kernel limit before the reconnect behavior is exercised.
        # The immutable conformance tree is read-only, so the socket belongs
        # in the supervisor-provided scratch mount rather than beside this
        # sealed test file.
        server.socket_path = socket_root / "s"
        server.close = types.MethodType(lambda self: None, server)
        executing = threading.Event()
        release = threading.Event()
        original_execute = server._execute

        def execute(self, operation, arguments):
            executing.set()
            assert release.wait(timeout=2)
            return original_execute(operation, arguments)

        server._execute = types.MethodType(execute, server)
        completed: list[int] = []
        failures: list[BaseException] = []

        def run_server():
            try:
                completed.append(server.serve())
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=run_server)
        thread.start()
        deadline = time.monotonic() + 2
        while (
            not server.socket_path.exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert server.socket_path.exists(), failures

        request_id = str(uuid.uuid4())
        request = _request(
            sequence=1,
            mutation_sequence=1,
            request_id=request_id,
        )
        wire = Worker._canonical_json(request) + b"\n"
        first = _connect_when_listening(server.socket_path)
        _socket_record(first)
        first.sendall(wire)
        assert executing.wait(timeout=2)
        first.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
        first.close()

        second = _connect_when_listening(server.socket_path)
        release.set()
        challenge = _socket_record(second)
        assert challenge["challenge_nonce"] == CHALLENGE_NONCE
        second.sendall(wire)
        response = _socket_record(second)
        assert response["request_id"] == request_id
        assert response["success"] is True
        second.close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert failures == []
        assert completed == [0]
        assert server.executions == 1
