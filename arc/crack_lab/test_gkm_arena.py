"""Unit checks for replay-compatible arena action encoding."""

import time

import pytest

import gkm_arena as A


class _Budget:
    def __init__(self):
        self.ticks = 0

    def tick(self):
        self.ticks += 1


class _FrameData:
    available_actions = (6,)
    levels_completed = 0
    state = "NOT_FINISHED"
    frame = [[[0]]]


class _Game:
    def __init__(self):
        self.inputs = []

    def perform_action(self, action, raw=True):
        self.inputs.append(action)
        return _FrameData()


def _arena():
    env = object.__new__(A.Arena)
    env._budget = _Budget()
    env._game = _Game()
    env._fd = _FrameData()
    env.path = []
    return env


def test_coordinate_action_is_forwarded_and_recorded_for_replay():
    env = _arena()
    env.step(6, 12, 34)
    assert env._game.inputs[-1].data == {"x": 12, "y": 34}
    assert env.path == [[6, 12, 34]]
    assert env.actions == (6,)


def test_coordinate_replay_token_and_integer_action_remain_supported():
    env = _arena()
    env.step([6, 7, 9])
    env.step(1)
    assert env.path == [[6, 7, 9], 1]


def test_coordinate_boundaries_are_public_screen_positions():
    env = _arena()
    env.step(6, 0, 0)
    env.step([6, 63, 63])
    assert env.path == [[6, 0, 0], [6, 63, 63]]


@pytest.mark.parametrize(
    "args",
    [
        (6,),
        (6, -1, 0),
        (6, 64, 0),
        (6, 0, 64),
        (6, True, 0),
        (6, 1.0, 0),
        ([6, -1, 0],),
        ([6, 0, 64],),
        ([6, False, 2],),
        ([6, 1, 2], 3, 4),
        (1, 0, 0),
        (True,),
        (1.0,),
        ("1",),
    ],
)
def test_invalid_public_actions_fail_before_budget_or_engine(args):
    env = _arena()
    with pytest.raises(ValueError):
        env.step(*args)
    assert env._budget.ticks == 0
    assert env._game.inputs == []
    assert env.path == []


def test_invalid_public_action_emits_protected_transcript_marker(capfd):
    env = _arena()
    with pytest.raises(A.PublicActionProtocolViolation):
        env.step(6, 64, 0)
    captured = capfd.readouterr()
    assert (
        f"{A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER}: "
        "coordinate action requires integer x,y in 0..63"
    ) in captured.err


def test_caught_clone_protocol_violation_poisoned_whole_program(monkeypatch):
    class FakeArena:
        def __init__(self, game, _budget=None):
            self._budget = _budget
            self.levels_completed = 0
            self.path = []

        def step(self, action, x=None, y=None):
            if x is not None and x >= 64:
                self._budget.protocol_violation = "x outside 0..63"
                raise A.PublicActionProtocolViolation("x outside 0..63")

    monkeypatch.setattr(A, "Arena", FakeArena)

    def catches_violation(env):
        try:
            env.step(6, 112, 18)
        except A.PublicActionProtocolViolation:
            pass

    levels, path, err = A.run_program(
        "fake", catches_violation, step_cap=10, time_cap=1
    )
    assert (levels, path) == (0, [])
    assert err == (
        f"{A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER}: x outside 0..63"
    )


def test_public_undo_action_is_forwarded_and_recorded():
    env = _arena()
    env._fd.available_actions = (6, 7)
    env.step(7)
    assert env._game.inputs[-1].data == {}
    assert env.path == [7]
    assert env.actions == (6,)


def test_run_program_wall_cap_interrupts_clone_only_computation(monkeypatch):
    class FakeArena:
        def __init__(self, game, _budget=None):
            self.levels_completed = 0
            self.path = []

        def step(self, action, x=None, y=None):
            self.path.append(action)

        def step(self, action, x=None, y=None):
            self.path.append(action)

    monkeypatch.setattr(A, "Arena", FakeArena)

    def never_steps(_env):
        while True:
            pass

    started = time.monotonic()
    levels, path, err = A.run_program(
        "fake", never_steps, step_cap=10, time_cap=0.05
    )
    assert time.monotonic() - started < 1
    assert levels == 0
    assert path == []
    assert err == "TimeoutError: program wall-time cap"


def test_proposer_interrupt_is_sanitized_without_internal_traceback(monkeypatch):
    class FakeArena:
        def __init__(self, game, _budget=None):
            self.levels_completed = 0
            self.path = []

        def step(self, action, x=None, y=None):
            self.path.append(action)

    monkeypatch.setattr(A, "Arena", FakeArena)
    monkeypatch.setenv("GKM_SANITIZE_PROPOSER_INTERRUPTS", "1")
    levels, path, err = A.run_program(
        "fake", lambda env: (_ for _ in ()).throw(KeyboardInterrupt()),
        time_cap=1,
    )
    assert (levels, path) == (0, [])
    assert err == "KeyboardInterrupt: proposer probe interrupted"


def test_operator_interrupt_still_propagates(monkeypatch):
    class FakeArena:
        def __init__(self, game, _budget=None):
            self.levels_completed = 0
            self.path = []

        def step(self, action, x=None, y=None):
            self.path.append(action)

    monkeypatch.setattr(A, "Arena", FakeArena)
    monkeypatch.delenv("GKM_SANITIZE_PROPOSER_INTERRUPTS", raising=False)
    with pytest.raises(KeyboardInterrupt):
        A.run_program(
            "fake", lambda env: (_ for _ in ()).throw(KeyboardInterrupt()),
            time_cap=1,
        )
