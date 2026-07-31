"""Search for a direct finish from level 5's final staged-cargo state."""

import gkm_try

from perception import bounded_bfs
from probe5_structure import PHASES, summary


class ReachedLevel5(Exception):
    pass


class StopAtLevel5:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 4:
            raise ReachedLevel5
        return result


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel5(env))
    except ReachedLevel5:
        pass
    start = env.clone()
    turn = 0
    for _, actions in PHASES:
        for action in actions:
            start.step(action)
            turn += 1
    base_level = start.levels_completed
    print("L5_TAIL_START", summary(start, turn), flush=True)
    path = bounded_bfs(
        start,
        lambda node, _path: node.levels_completed > base_level,
        max_states=8000,
        max_depth=13,
    )
    print("L5_TAIL_PATH", path, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
