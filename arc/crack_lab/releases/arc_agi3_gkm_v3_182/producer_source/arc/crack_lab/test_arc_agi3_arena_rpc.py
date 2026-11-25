from __future__ import annotations

import copy
import json
import os
import socket
import sys
import threading
import uuid
from pathlib import Path

import pytest

import arc_agi3_arena_rpc as Rpc
import arc_agi3_contiguous_conformance as Conformance
import arc_agi3_contiguous_supervisor as Supervisor
import arc_agi3_container_worker as Worker


@pytest.fixture
def rpc_socket_path() -> Path:
    root = Supervisor._private_system_scratch()
    metadata = root.stat(follow_symlinks=False)
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
    )
    path = root / "s"
    assert len(os.fsencode(path)) < 104
    try:
        yield path
    finally:
        Conformance._remove_owned_private_tree(
            root,
            expected_identity=identity,
            label="Arena RPC test scratch",
        )


def test_rpc_socket_root_is_short_from_overlong_working_directory(
    tmp_path: Path, monkeypatch, rpc_socket_path: Path,
):
    long_cwd = tmp_path
    while len(os.fsencode(long_cwd)) <= 120:
        long_cwd = long_cwd / ("long-working-directory-" + "x" * 20)
    long_cwd.mkdir(parents=True)
    monkeypatch.chdir(long_cwd)
    assert len(os.fsencode(Path.cwd())) > 104
    assert len(os.fsencode(rpc_socket_path)) < 104
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(rpc_socket_path))
        assert rpc_socket_path.exists()
    finally:
        listener.close()
        if rpc_socket_path.exists():
            rpc_socket_path.unlink()


class FakeArena:
    created: list["FakeArena"] = []

    def __init__(self, game: str):
        self.game = game
        self.value = 0
        self.path: list[object] = []
        self._levels = 0
        self.reset_calls = 0
        type(self).created.append(self)

    @property
    def actions(self):
        return (1, 2, 6, 7)

    @property
    def levels_completed(self):
        return self._levels

    def terminal(self):
        return len(self.path) >= 6

    def frame(self):
        return [[self.value, 0], [0, self._levels]]

    def reset(self):
        self.reset_calls += 1
        self.value = 0
        self.path = []
        self._levels = 0
        return self.frame()

    def step(self, action, x=None, y=None):
        if action == 6:
            self.value = int(x) + int(y)
            self.path.append([6, int(x), int(y)])
        else:
            self.value += int(action)
            self.path.append(int(action))
        if self.value >= 5:
            self._levels = 1
        return self.frame()

    def clone(self):
        cloned = copy.deepcopy(self)
        type(self).created.append(cloned)
        return cloned


class StickyRewardUndoArena:
    """ACTION7 cannot undo the already-issued public reward transition."""

    created: list["StickyRewardUndoArena"] = []

    def __init__(self, game: str):
        self.game = game
        self.value = 0
        self.path: list[object] = []
        self._levels = 0
        self.undo_calls = 0
        type(self).created.append(self)

    @property
    def actions(self):
        return (1, 2, 7)

    @property
    def levels_completed(self):
        return self._levels

    def terminal(self):
        return False

    def frame(self):
        return [[self.value, self._levels]]

    def reset(self):
        raise AssertionError("the controller must replace, not reset")

    def step(self, action, x=None, y=None):
        assert x is None
        assert y is None
        action = int(action)
        if action == 7:
            self.undo_calls += 1
            # This models the observed public edge: ordinary probes can be
            # restored, but a rewarded transition remains post-reward.
            if self._levels == 0:
                self.value = 0
        elif action == 2:
            self.value = 9
            self._levels = 1
        else:
            self.value += action
        self.path.append(action)
        return self.frame()

    def clone(self):
        cloned = copy.deepcopy(self)
        type(self).created.append(cloned)
        return cloned


class ContextSpecificUndoArena:
    """One ordinary action has an exact public ACTION7 inverse."""

    mismatch = False

    def __init__(self, game: str):
        self.game = game
        self.value = 0
        self.path: list[object] = []

    @property
    def actions(self):
        return (1, 7)

    @property
    def levels_completed(self):
        return 0

    def terminal(self):
        return False

    def frame(self):
        return [[self.value, 0]]

    def reset(self):
        raise AssertionError("the controller must replace, not reset")

    def step(self, action, x=None, y=None):
        assert x is None
        assert y is None
        action = int(action)
        if action == 1:
            self.value = 1
        elif action == 7:
            self.value = 2 if self.mismatch else 0
        self.path.append(action)
        return self.frame()

    def clone(self):
        return copy.deepcopy(self)


class MismatchedContextUndoArena(ContextSpecificUndoArena):
    mismatch = True


class SiblingLeakyArena:
    """Clones have distinct paths but share gameplay state by reference."""

    def __init__(self, game: str):
        self.game = game
        self.path: list[object] = []
        self._shared = {"value": 0}

    @property
    def actions(self):
        return (1, 2)

    @property
    def levels_completed(self):
        return 1 if self._shared["value"] >= 5 else 0

    def terminal(self):
        return False

    def frame(self):
        return [[self._shared["value"], self.levels_completed]]

    def reset(self):
        raise AssertionError("the controller must replace, not reset")

    def step(self, action, x=None, y=None):
        assert x is None
        assert y is None
        self._shared["value"] += int(action)
        self.path.append(int(action))
        return self.frame()

    def clone(self):
        cloned = copy.copy(self)
        cloned.path = list(self.path)
        # Deliberate defect: ``_shared`` remains aliased.
        return cloned


class ExhaustedParentArena:
    """Fake game with a 600-action L1 parent and a fresh 600-action L2 path."""

    created = []

    def __init__(self, game):
        self.game = game
        self.path = []
        self.reset_calls = 0
        type(self).created.append(self)

    @property
    def actions(self):
        return (1, 2)

    @property
    def levels_completed(self):
        if self.path and all(action == 2 for action in self.path):
            if len(self.path) >= 600:
                return 2
            if len(self.path) >= 300:
                return 1
            return 0
        return 1 if len(self.path) >= 600 else 0

    def terminal(self):
        return len(self.path) >= 600

    def frame(self):
        return [[len(self.path) % 16, self.levels_completed]]

    def reset(self):
        self.reset_calls += 1
        self.path = []
        return self.frame()

    def step(self, action, x=None, y=None):
        assert x is None
        assert y is None
        self.path.append(int(action))
        return self.frame()

    def clone(self):
        cloned = copy.deepcopy(self)
        type(self).created.append(cloned)
        return cloned


def binding(
    *,
    parent_level: int = 0,
    target_level: int | None = None,
    exploration_mode: str = "continue_parent",
) -> Rpc.ArenaSessionBinding:
    return Rpc.ArenaSessionBinding(
        campaign_id="campaign-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
        game="zz99",
        parent_level=parent_level,
        target_level=(
            parent_level + 1
            if target_level is None
            else target_level
        ),
        parent_checkpoint_sha256="a" * 64,
        frontier_sha256="b" * 64,
        exploration_mode=exploration_mode,
    )


def session(
    *,
    parent_level: int = 0,
    parent_path: tuple[object, ...] = (),
    token: str = "t" * 64,
    real_step_cap: int = 6,
    total_step_cap: int = 40,
    reset_cap: int = 8,
    arena_factory=FakeArena,
    exploration_mode: str = "continue_parent",
) -> Rpc.ArenaHostSession:
    FakeArena.created = []
    return Rpc.ArenaHostSession(
        "zz99",
        binding=binding(
            parent_level=parent_level,
            exploration_mode=exploration_mode,
        ),
        parent_path=parent_path,
        arena_factory=arena_factory,
        real_step_cap=real_step_cap,
        total_step_cap=total_step_cap,
        reset_cap=reset_cap,
        token=token,
    )


def rpc_request(
    host: Rpc.ArenaHostSession,
    seq: int,
    op: str,
    **fields,
):
    unsigned = {
        "schema": Rpc.RPC_SCHEMA,
        "session": host.session_id,
        "seq": seq,
        "op": op,
        **fields,
    }
    return {
        **unsigned,
        "mac": Rpc._wire_mac(host.token, unsigned),
    }


@pytest.fixture
def running_server(tmp_path: Path, rpc_socket_path: Path):
    host = session()
    socket_path = rpc_socket_path
    transcript = tmp_path / "host" / "arena.jsonl"
    server = Rpc.ArenaRpcServer(host, socket_path, transcript)
    thread = server.start_thread()
    yield host, server, thread, socket_path, transcript
    if thread.is_alive():
        try:
            Rpc.ArenaRpcClient(socket_path, host.token).close()
        except (OSError, Rpc.ArenaRpcError):
            pass
    thread.join(timeout=5)


def test_parent_is_replayed_before_first_observation_and_bound():
    host = session(
        parent_level=1,
        parent_path=(2, 2, 1),
        real_step_cap=6,
    )
    assert host.binding.parent_level == 1
    assert host._parent_path == (2, 2, 1)
    assert host._seed_snapshot["levels_completed"] == 1
    assert host._parent_replay_steps == 3
    assert host._total_steps == 3
    assert host.binding_event()["parent_path_sha256"] == (
        Rpc.hashlib.sha256(
            Rpc._canonical_json([2, 2, 1])
        ).hexdigest()
    )


def test_nonzero_parent_cannot_start_from_zero_or_wrong_boundary():
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="parent path",
    ):
        session(parent_level=1, parent_path=())

    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="parent level",
    ):
        session(parent_level=1, parent_path=(1,))

    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="exact",
    ):
        session(parent_level=1, parent_path=(7, 1))


def test_fresh_factory_must_be_public_zero_state():
    class NonzeroArena(FakeArena):
        def __init__(self, game: str):
            super().__init__(game)
            self.path = [1]

    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="zero state",
    ):
        session(arena_factory=NonzeroArena)


def test_clone_must_be_independent_and_publicly_identical():
    class SameCloneArena(FakeArena):
        def clone(self):
            return self

    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="independent",
    ):
        session(arena_factory=SameCloneArena)

    class DivergentCloneArena(FakeArena):
        def clone(self):
            cloned = copy.deepcopy(self)
            cloned.value += 1
            return cloned

    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="fresh-process",
    ):
        session(arena_factory=DivergentCloneArena)


def test_sibling_clone_leak_selects_authenticated_fresh_process():
    host = session(arena_factory=SiblingLeakyArena)
    event = host.binding_event()
    assert (
        event["probe_isolation_mode"]
        == Rpc.ProbeContract.FRESH_PROCESS_PER_CANDIDATE_MODE
    )
    mode, digest = (
        Rpc.ProbeContract.validate_probe_isolation_evidence(
            event["probe_isolation_evidence"],
            expected_seed_snapshot_sha256=event[
                "exploration_seed_snapshot_sha256"
            ],
            expected_seed_path_sha256=event[
                "exploration_seed_path_sha256"
            ],
        )
    )
    assert mode == event["probe_isolation_mode"]
    assert digest == event["probe_isolation_evidence_sha256"]
    assert event["probe_isolation_evidence"]["canary_status"] == "LEAK"
    assert (
        event["probe_isolation_evidence"]["mutable_graph_status"]
        == "LEAK"
    )
    assert (
        event["probe_isolation_evidence"][
            "shared_mutable_identity_count"
        ]
        > 0
    )

    first_identity = tuple(host._fresh_process_identity_sha256s)
    host.dispatch(rpc_request(host, 0, "open"))
    host.dispatch(rpc_request(host, 1, "step", action=1))
    host.dispatch(rpc_request(host, 2, "reset"))
    assert len(host._fresh_process_identity_sha256s) == 2
    assert (
        tuple(host._fresh_process_identity_sha256s)[:1]
        == first_identity
    )
    assert len(set(host._fresh_process_identity_sha256s)) == 2
    host.dispatch(rpc_request(host, 3, "step", action=2))
    response = host.dispatch(rpc_request(host, 4, "close"))
    host._mark_close_delivered(response["seq"])
    result = host.host_result()
    assert result.path == (2,)
    assert result.resets == 1

    second = session(arena_factory=SiblingLeakyArena)
    forged = rpc_request(second, 0, "open")
    forged["probe_isolation_mode"] = (
        Rpc.ProbeContract.VERIFIED_ISOLATED_CLONE_MODE
    )
    forged["mac"] = Rpc._wire_mac(
        second.token,
        {key: value for key, value in forged.items() if key != "mac"},
    )
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="fields mismatch",
    ):
        second.dispatch(forged)
    second._discard_resources()


def test_round_trip_uses_one_clone_and_reset_reclones_seed(
    running_server,
):
    host, server, thread, socket_path, transcript = running_server
    with Rpc.ArenaRpcClient(socket_path, host.token) as client:
        env = client.root
        assert env.actions == (1, 2, 6, 7)
        assert env.levels_completed == 0
        assert not hasattr(env, "clone")
        env.step(2)
        env.step(2)
        assert int(env.frame()[0][0]) == 4
        env.reset()
        assert int(env.frame()[0][0]) == 0
        assert env.levels_completed == 0
        env.step(1)

    thread.join(timeout=5)
    server.wait(1)
    result = host.host_result()
    assert result.game == "zz99"
    assert result.parent_path == ()
    assert result.path == (1,)
    assert result.parent_replay_steps == 0
    assert result.exploration_steps == 3
    assert result.resets == 1
    assert result.total_steps == 3
    assert all(arena.reset_calls == 0 for arena in FakeArena.created)

    events = [
        json.loads(line)
        for line in transcript.read_text().splitlines()
    ]
    assert events[0]["kind"] == "arena_session_binding"
    assert events[0]["binding_sha256"] == host.binding_sha256
    assert all("token" not in event for event in events)
    text = transcript.read_text()
    assert host.token not in text
    assert '"frame"' not in text
    assert '"parent_path":' not in text


def test_reward_boundary_is_absorbing_before_action7_undo():
    StickyRewardUndoArena.created = []
    host = session(
        arena_factory=StickyRewardUndoArena,
        real_step_cap=6,
    )
    assert (
        host.binding_event()["reward_boundary_policy"]
        == Rpc.REWARD_BOUNDARY_POLICY
    )
    host.dispatch(rpc_request(host, 0, "open"))
    rewarded = host.dispatch(
        rpc_request(host, 1, "step", action=2)
    )
    assert rewarded["result"]["snapshot"]["levels_completed"] == 1
    assert host._reward_boundary is not None
    assert Rpc._SHA256_RE.fullmatch(
        host._reward_boundary_sha256
    )

    for seq, operation, fields in (
        (2, "step", {"action": 7}),
        (3, "reset", {}),
        (4, "observe", {}),
    ):
        with pytest.raises(
            Rpc.ArenaRpcContractError,
            match="reward boundary is sealed",
        ):
            host.dispatch(
                rpc_request(host, seq, operation, **fields)
            )

    assert host._exploration.path == [2]
    assert host._exploration.levels_completed == 1
    assert host._exploration.undo_calls == 0
    closed = host.dispatch(rpc_request(host, 5, "close"))
    host._mark_close_delivered(closed["seq"])
    result = host.host_result()
    assert result.path == (2,)
    assert result.levels_completed == 1


def test_action7_requires_exact_context_restoration():
    host = session(
        arena_factory=ContextSpecificUndoArena,
        real_step_cap=6,
    )
    assert (
        host.binding_event()["action7_rollback_policy"]
        == Rpc.ACTION7_ROLLBACK_POLICY
    )
    opened = host.dispatch(rpc_request(host, 0, "open"))
    initial = opened["result"]["snapshot"]
    host.dispatch(rpc_request(host, 1, "step", action=1))
    rolled_back = host.dispatch(
        rpc_request(host, 2, "step", action=7)
    )
    assert rolled_back["result"]["snapshot"]["frame"] == (
        initial["frame"]
    )
    assert (
        rolled_back["result"]["snapshot"]["levels_completed"]
        == initial["levels_completed"]
    )
    assert (
        rolled_back["result"]["snapshot"]["terminal"]
        == initial["terminal"]
    )
    assert host._branch_invalidated is False
    assert host._rollback_reconstructions == 0
    closed = host.dispatch(rpc_request(host, 3, "close"))
    host._mark_close_delivered(closed["seq"])
    result = host.host_result()
    assert result.path == (1, 7)
    assert result.levels_completed == 0


def test_action7_frame_mismatch_invalidates_and_reconstructs_branch():
    host = session(
        arena_factory=MismatchedContextUndoArena,
        real_step_cap=6,
    )
    opened = host.dispatch(rpc_request(host, 0, "open"))
    initial = opened["result"]["snapshot"]
    host.dispatch(rpc_request(host, 1, "step", action=1))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match=(
            "failed exact frame/level/terminal rollback; "
            "exploration branch was invalidated and reconstructed"
        ),
    ):
        host.dispatch(rpc_request(host, 2, "step", action=7))

    assert host._branch_invalidated is True
    assert host._rollback_reconstructions == 1
    assert host._exploration.path == []
    assert Rpc._snapshot_payload(host._exploration) == initial
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="invalidated",
    ):
        host.dispatch(rpc_request(host, 3, "observe"))
    closed = host.dispatch(rpc_request(host, 4, "close"))
    host._mark_close_delivered(closed["seq"])
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="lineage or step accounting",
    ):
        host.host_result()


def test_parent_seed_remains_immutable_across_step_and_reset():
    host = session(parent_level=1, parent_path=(2, 2, 1))
    host.dispatch(rpc_request(host, 0, "open"))
    host.dispatch(rpc_request(host, 1, "step", action=1))
    host.dispatch(rpc_request(host, 2, "reset"))
    assert host._seeded_root.path == [2, 2, 1]
    assert host._seeded_root.reset_calls == 0
    assert host._exploration.path == [2, 2, 1]


def test_seed_or_clone_out_of_band_mutation_fails_closed():
    host = session()
    host.dispatch(rpc_request(host, 0, "open"))
    host._seeded_root.path.append(1)
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="seeded",
    ):
        host.dispatch(rpc_request(host, 1, "observe"))

    host = session()
    host.dispatch(rpc_request(host, 0, "open"))
    host._exploration.path.append(1)
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="outside",
    ):
        host.dispatch(rpc_request(host, 1, "observe"))


def test_parent_budget_and_exploration_budget_are_separate():
    host = session(
        parent_level=1,
        parent_path=(2, 2, 1),
        real_step_cap=4,
        total_step_cap=5,
    )
    host.dispatch(rpc_request(host, 0, "open"))
    host.dispatch(rpc_request(host, 1, "step", action=1))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="600-action",
    ):
        host.dispatch(rpc_request(host, 2, "step", action=1))

    host = session(real_step_cap=2, total_step_cap=2)
    host.dispatch(rpc_request(host, 0, "open"))
    host.dispatch(rpc_request(host, 1, "step", action=1))
    host.dispatch(rpc_request(host, 2, "reset"))
    host.dispatch(rpc_request(host, 3, "step", action=1))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="total exploration",
    ):
        host.dispatch(rpc_request(host, 4, "step", action=1))


def test_exhausted_parent_requires_explicit_fresh_prefix_mode():
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="requires fresh-prefix",
    ):
        session(
            parent_level=1,
            parent_path=(1,) * 600,
            real_step_cap=600,
            total_step_cap=2_000,
            arena_factory=ExhaustedParentArena,
        )


def test_fresh_prefix_replays_exhausted_parent_but_explores_from_zero():
    host = session(
        parent_level=1,
        parent_path=(1,) * 600,
        real_step_cap=600,
        total_step_cap=2_000,
        arena_factory=ExhaustedParentArena,
        exploration_mode="fresh_prefix",
    )
    event = host.binding_event()
    assert event["exploration_mode"] == "fresh_prefix"
    assert host._seeded_root.path == [1] * 600
    assert host._exploration.path == []

    host.dispatch(rpc_request(host, 0, "open"))
    for seq in range(1, 601):
        host.dispatch(rpc_request(host, seq, "step", action=2))
    host.dispatch(rpc_request(host, 601, "close"))
    host._mark_close_delivered(601)

    result = host.host_result()
    assert result.exploration_mode == "fresh_prefix"
    assert result.parent_level == 1
    assert result.levels_completed == 2
    assert result.parent_path == (1,) * 600
    assert result.path == (2,) * 600
    assert result.parent_replay_steps == 600
    assert result.exploration_steps == 600
    assert result.total_steps == 1_200
    assert all(arena.reset_calls == 0 for arena in ExhaustedParentArena.created)


@pytest.mark.parametrize(
    ("parent_level", "parent_length"),
    [(0, 0), (1, 599)],
)
def test_fresh_prefix_rejects_nonexhausted_parent(
    parent_level,
    parent_length,
):
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="fresh-prefix",
    ):
        session(
            parent_level=parent_level,
            parent_path=(1,) * parent_length,
            real_step_cap=600,
            total_step_cap=2_000,
            arena_factory=ExhaustedParentArena,
            exploration_mode="fresh_prefix",
        )


def test_fresh_prefix_total_cap_funds_parent_and_complete_candidate():
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="fund one complete",
    ):
        session(
            parent_level=1,
            parent_path=(1,) * 600,
            real_step_cap=600,
            total_step_cap=1_199,
            arena_factory=ExhaustedParentArena,
            exploration_mode="fresh_prefix",
        )


def test_reset_budget_is_bounded_without_calling_engine_reset():
    host = session(reset_cap=1)
    host.dispatch(rpc_request(host, 0, "open"))
    host.dispatch(rpc_request(host, 1, "reset"))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="reset budget",
    ):
        host.dispatch(rpc_request(host, 2, "reset"))
    assert all(arena.reset_calls == 0 for arena in FakeArena.created)


def test_hmac_session_and_sequence_are_mandatory():
    host = session()
    request = rpc_request(host, 0, "open")
    assert "token" not in request
    forged = dict(request)
    forged["mac"] = "0" * 64
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="authentication",
    ):
        host.dispatch(forged)

    host = session()
    opened = host.dispatch(rpc_request(host, 0, "open"))
    assert opened["ok"] is True
    assert opened["session"] == host.session_id
    assert Rpc._SHA256_RE.fullmatch(opened["mac"])
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="stale",
    ):
        host.dispatch(rpc_request(host, 0, "observe"))


def test_only_frozen_worker_operations_are_accepted():
    assert Rpc._PUBLIC_OPERATIONS == {
        "open",
        "observe",
        "reset",
        "step",
        "close",
    }
    host = session()
    host.dispatch(rpc_request(host, 0, "open"))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="unknown",
    ):
        host.dispatch(rpc_request(host, 1, "clone"))


def test_host_result_requires_durably_delivered_close():
    host = session()
    host.dispatch(rpc_request(host, 0, "open"))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="clean session close",
    ):
        host.host_result()
    host.dispatch(rpc_request(host, 1, "close"))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="clean session close",
    ):
        host.host_result()
    host._mark_close_delivered(1)
    assert host.host_result().path == ()


def test_client_returns_frame_copy_and_observe_refreshes(
    running_server,
):
    host, _server, _thread, socket_path, _transcript = running_server
    with Rpc.ArenaRpcClient(socket_path, host.token) as client:
        frame = client.root.frame()
        frame[0][0] = 15
        assert int(client.root.frame()[0][0]) == 0
        client.root.observe()
        assert int(client.root.frame()[0][0]) == 0


def test_out_of_range_client_call_is_host_logged_protocol_invalid(
    running_server,
):
    host, server, thread, socket_path, transcript = running_server
    client = Rpc.ArenaRpcClient(socket_path, host.token)

    with pytest.raises(
        Rpc.ArenaRpcError,
        match=r"coordinate action must be \[6, x, y\] with x,y in 0\.\.63",
    ):
        client.root.step(6, 18, 112)

    thread.join(timeout=5)
    server.wait(1)
    assert client._closed is True
    assert host._exploration.path == []
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="clean session close",
    ):
        host.host_result()

    events = [
        json.loads(line)
        for line in transcript.read_text().splitlines()
    ]
    rejected = [
        event
        for event in events
        if event.get("kind") == "rpc"
        and event.get("phase") == "rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["op"] == "step"
    assert rejected[0]["ok"] is False
    assert "0..63" in rejected[0]["error"]
    assert not any(
        event.get("kind") == "rpc"
        and event.get("phase") == "applied"
        and event.get("op") == "step"
        for event in events
    )


def test_protocol_invalid_callback_precedes_catchable_error(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session(token="p" * 64)
    socket_path = rpc_socket_path
    transcript = tmp_path / "host.jsonl"
    server = Rpc.ArenaRpcServer(host, socket_path, transcript)
    callback_events: list[dict] = []

    def invalidate(event):
        # The rejected event must already be durable when containment begins.
        rows = [
            json.loads(line)
            for line in transcript.read_text().splitlines()
        ]
        assert rows[-1] == event
        callback_events.append(dict(event))

    server.set_protocol_violation_callback(invalidate)
    thread = server.start_thread()
    client = Rpc.ArenaRpcClient(socket_path, host.token)
    with pytest.raises(Rpc.ArenaRpcError, match="0..63"):
        client.root.step(6, 18, 112)
    thread.join(timeout=5)
    server.wait(1)
    assert len(callback_events) == 1
    assert callback_events[0]["phase"] == "rejected"
    assert server.protocol_violation == callback_events[0]
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="exactly once",
    ):
        server.set_protocol_violation_callback(invalidate)


@pytest.mark.parametrize(
    "action",
    [
        True,
        0,
        6,
        8,
        [6, 1],
        [5, 1, 2],
        [6, -1, 0],
        [6, 0, 64],
        [6, True, 2],
        {"action": 1},
    ],
)
def test_action_grammar_rejects_aliases_and_out_of_range(action):
    with pytest.raises(Rpc.ArenaRpcContractError):
        Rpc._normalize_action(action)


def test_engine_coordinate_tuples_and_wire_lists_normalize_identically():
    assert Rpc._normalize_action([6, 3, 4]) == (6, 3, 4)
    assert Rpc._normalize_action((6, 3, 4)) == (6, 3, 4)


def test_unavailable_actions_and_duplicate_actions_are_rejected():
    host = session()
    host.dispatch(rpc_request(host, 0, "open"))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="not currently available",
    ):
        host.dispatch(rpc_request(host, 1, "step", action=3))
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="unique",
    ):
        Rpc._normalize_actions([1, 1])


@pytest.mark.parametrize("cell", [True, 1.0, "1", None])
def test_frame_schema_rejects_non_integral_aliases(cell):
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="integer",
    ):
        Rpc._normalize_frame([[cell]])


def test_frame_schema_rejects_ragged_rows():
    with pytest.raises(Rpc.ArenaRpcContractError):
        Rpc._normalize_frame([[0, 1], [2]])


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"x","schema":"y"}',
        b'{"seq":NaN}',
        b'{"seq":Infinity}',
    ],
)
def test_json_decoder_rejects_duplicate_and_nonfinite_aliases(raw):
    with pytest.raises(Rpc.ArenaRpcContractError):
        Rpc._loads_json(raw, label="test")


def test_framing_rejects_partial_eof_and_pipelining():
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(b"{}")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(
            Rpc.ArenaRpcContractError,
            match="newline",
        ):
            Rpc._recv_line(receiver)
    finally:
        receiver.close()
        sender.close()

    receiver, sender = socket.socketpair()
    try:
        sender.sendall(b"{}\n{}\n")
        with pytest.raises(
            Rpc.ArenaRpcContractError,
            match="pipelined",
        ):
            Rpc._recv_line(receiver)
    finally:
        receiver.close()
        sender.close()


@pytest.mark.parametrize(
    "token",
    ["", "x" * 31, "x" * 64 + "\n"],
)
def test_session_rejects_token_aliases(token):
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="token",
    ):
        session(token=token)


def test_binding_rejects_wrong_frontier_and_hashes():
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="binding",
    ):
        Rpc.ArenaHostSession(
            "zz99",
            binding=Rpc.ArenaSessionBinding(
                **{
                    **Rpc.asdict(binding()),
                    "target_level": 3,
                }
            ),
            parent_path=(),
            arena_factory=FakeArena,
        )


def test_unknown_operation_is_sanitized_and_closes_transport(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session(token="x" * 64)
    socket_path = rpc_socket_path
    transcript = tmp_path / "host.jsonl"
    server = Rpc.ArenaRpcServer(host, socket_path, transcript)
    thread = server.start_thread()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    reader = client.makefile("rb")
    client.sendall(
        Rpc._canonical_json(rpc_request(host, 0, "open")) + b"\n"
    )
    assert json.loads(reader.readline())["ok"] is True
    client.sendall(
        Rpc._canonical_json(
            rpc_request(host, 1, host.token)
        )
        + b"\n"
    )
    response = json.loads(reader.readline())
    reader.close()
    client.close()
    thread.join(timeout=5)
    server.wait(1)
    assert response["ok"] is False
    assert response["error"] == "unknown Arena RPC operation"
    assert Rpc._SHA256_RE.fullmatch(response["mac"])
    transcript_text = transcript.read_text()
    assert host.token not in transcript_text
    assert "Traceback" not in transcript_text
    assert "FakeArena" not in transcript_text


def test_listener_accepts_only_one_client(running_server):
    host, _server, _thread, socket_path, _transcript = running_server
    client = Rpc.ArenaRpcClient(socket_path, host.token)
    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(OSError):
        second.connect(str(socket_path))
    second.close()
    client.close()


def test_client_rejects_forged_response_hmac():
    client_socket, server_socket = socket.socketpair()
    client = object.__new__(Rpc.ArenaRpcClient)
    client._socket = client_socket
    client._token = "t" * 64
    client._session_id = Rpc._wire_mac(
        client._token,
        {"schema": Rpc.RPC_SCHEMA, "kind": "session"},
    )
    client._seq = 0
    client._lock = threading.RLock()
    client._closed = False

    response = {
        "schema": Rpc.RPC_SCHEMA,
        "session": client._session_id,
        "seq": 0,
        "ok": True,
        "result": {
            "binding_sha256": "a" * 64,
            "snapshot": {
                "frame": [[0]],
                "actions": [1],
                "levels_completed": 0,
                "terminal": False,
            },
        },
        "mac": "0" * 64,
    }

    def respond():
        assert Rpc._recv_line(server_socket) is not None
        Rpc._send_json(server_socket, response)

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        with pytest.raises(Rpc.ArenaRpcError, match="HMAC"):
            client._call("open")
    finally:
        client._closed = True
        client._token = ""
        client_socket.close()
        server_socket.close()
        thread.join(timeout=2)


def test_host_transcript_rejects_symlink(tmp_path: Path):
    target = tmp_path / "outside"
    target.write_text("unchanged")
    link = tmp_path / "transcript"
    link.symlink_to(target)
    with pytest.raises(Rpc.ArenaRpcContractError):
        Rpc.HostTranscript(link)
    assert target.read_text() == "unchanged"


def test_host_transcript_detects_hardlink_and_replacement(
    tmp_path: Path,
):
    path = tmp_path / "transcript"
    alias = tmp_path / "alias"
    transcript = Rpc.HostTranscript(path)
    os.link(path, alias)
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="custody",
    ):
        transcript.append({"kind": "test"})
    alias.unlink()
    transcript.close()

    path = tmp_path / "second"
    transcript = Rpc.HostTranscript(path)
    path.unlink()
    path.write_text("replacement")
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="custody",
    ):
        transcript.close()
    assert path.read_text() == "replacement"


def test_host_transcript_detects_parent_replacement(tmp_path: Path):
    parent = tmp_path / "host"
    path = parent / "transcript"
    parent.mkdir()
    transcript = Rpc.HostTranscript(path)
    parent.rename(tmp_path / "moved-host")
    parent.mkdir()
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="custody",
    ):
        transcript.append({"kind": "test"})
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="custody",
    ):
        transcript.close()


def test_host_transcript_rejects_nested_forbidden_values(
    tmp_path: Path,
):
    secret = "s" * 64
    transcript = Rpc.HostTranscript(
        tmp_path / "transcript",
        forbidden_values=(secret,),
    )
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="forbidden",
    ):
        transcript.append(
            {"detail": {"value": f"prefix-{secret}-suffix"}}
        )
    transcript.close()
    assert secret not in (
        tmp_path / "transcript"
    ).read_text()


def test_server_shutdown_interrupts_idle_listener(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session()
    socket_path = rpc_socket_path
    server = Rpc.ArenaRpcServer(
        host,
        socket_path,
        tmp_path / "host.jsonl",
    )
    thread = server.start_thread()
    server.shutdown()
    thread.join(timeout=3)
    assert not thread.is_alive()
    server.wait(1)
    assert not socket_path.exists()
    with pytest.raises(
        Rpc.ArenaRpcContractError,
        match="clean session close",
    ):
        host.host_result()


def test_server_shutdown_interrupts_connected_client(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session()
    socket_path = rpc_socket_path
    server = Rpc.ArenaRpcServer(
        host,
        socket_path,
        tmp_path / "host.jsonl",
    )
    thread = server.start_thread()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    server.shutdown()
    thread.join(timeout=3)
    client.close()
    assert not thread.is_alive()
    server.wait(1)
    assert not socket_path.exists()


def test_socket_replacement_is_not_unlinked(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session()
    socket_path = rpc_socket_path
    server = Rpc.ArenaRpcServer(
        host,
        socket_path,
        tmp_path / "host.jsonl",
    )
    thread = server.start_thread()
    socket_path.unlink()
    socket_path.write_text("replacement")
    server.shutdown()
    thread.join(timeout=3)
    assert not thread.is_alive()
    with pytest.raises(Rpc.ArenaRpcError):
        server.wait(1)
    assert socket_path.read_text() == "replacement"
    socket_path.unlink()


def test_pinned_worker_uses_only_default_rpc_clone(
    tmp_path: Path, rpc_socket_path: Path,
):
    host = session(
        token="w" * 64,
        real_step_cap=6,
        total_step_cap=20,
    )
    socket_path = rpc_socket_path
    transcript = tmp_path / "host" / "rpc.jsonl"
    server = Rpc.ArenaRpcServer(host, socket_path, transcript)
    thread = server.start_thread()

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    solve = attempt / "solve.py"
    solve.write_text(
        "def solve(env):\n"
        "    env.step(2)\n"
        "    env.reset()\n"
        "    env.step(2)\n"
        "    env.step(2)\n"
        "    env.step(1)\n"
    )
    token = tmp_path / "token"
    token.write_text(host.token)
    config = Worker.WorkerConfig(
        socket_path=socket_path,
        token_file=token,
        solve_path=solve,
        outcome_path=tmp_path / "output" / "outcome.json",
    )
    original_path = list(sys.path)
    missing = object()
    original_modules = {
        name: sys.modules.get(name, missing)
        for name in ("solve", "players", "legs")
    }
    try:
        outcome = Worker.run_worker(config)
    finally:
        sys.path[:] = original_path
        for name, module in original_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    thread.join(timeout=5)
    server.wait(1)

    assert outcome["status"] == "completed"
    assert outcome["authoritative"] is False
    result = host.host_result()
    assert result.path == (2, 2, 1)
    assert result.parent_replay_steps == 0
    assert result.exploration_steps == 4
    assert result.resets == 1
    assert result.levels_completed == 1
