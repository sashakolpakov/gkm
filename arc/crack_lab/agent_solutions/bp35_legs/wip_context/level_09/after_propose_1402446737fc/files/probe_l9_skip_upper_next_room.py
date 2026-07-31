"""Test the two gravity-switch choices after the one-band corridor climb."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_shift
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip_upper_corridor import corridor
from probe_l9_skip_upper_endgame import objects


def step(env, action):
    before = env.frame()
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    return band_shift(before, env.frame())


def stage(root):
    child = corridor(root)
    step(child, (6, 15, 33))
    return child


def summary(env):
    return {
        "terminal": bool(env.terminal()),
        "level": int(env.levels_completed) + 1,
        "controls": controls(env),
        "grid": compact(env)["grid9"],
        "objects": objects(env),
    }


def branch(root, name, actions):
    child = stage(root)
    print(name, 0, summary(child), flush=True)
    for index, action in enumerate(actions, 1):
        gain = step(child, action)
        print(name, index, action, "gain", gain, summary(child), flush=True)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for which in (0, -1):
        staged = stage(env)
        switch = controls(staged)[which]
        branch(env, ("FLIP_MOVE", which), (switch, 3, 4, 4, 4))
        for click in (
            (6, 15, 33),
            (6, 21, 39),
            (6, 27, 39),
            (6, 15, 3),
            (6, 15, 57),
        ):
            branch(env, ("FLIP_CLICK", which, click), (switch, click, 3, 4))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
