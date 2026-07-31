"""Probe the upper corridor exposed by omitting one redundant climb."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_shift
from probe_l9_control_row import compact, controls
from probe_l9_skip_upper_endgame import objects, root_skip
from probe_l9_route_deletions import enter_level_9


def step(env, action):
    before = env.frame()
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    return band_shift(before, env.frame())


def report(label, env, gain=0):
    print(
        label,
        "gain",
        gain,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "state",
        compact(env),
        "objects",
        objects(env),
        flush=True,
    )


def corridor(root):
    child = root_skip(root)
    step(child, controls(child)[0])
    step(child, 4)
    step(child, 3)
    step(child, 3)
    return child


def trace(root, name, actions):
    child = corridor(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        gain = step(child, action)
        report((name, index, action), child, gain)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    climb = (6, 15, 33)
    trace(env, "CLIMB", (climb,) * 10)
    for action in (
        (6, 21, 33),
        (6, 27, 33),
        (6, 15, 45),
        (6, 15, 21),
        (6, 3, 57),
        4,
        7,
    ):
        trace(env, ("LOCAL", action), (action, 3, 4))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
