"""Unit checks for replay-compatible arena action encoding."""

import gkm_arena as A


class _Budget:
    def tick(self):
        pass


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
