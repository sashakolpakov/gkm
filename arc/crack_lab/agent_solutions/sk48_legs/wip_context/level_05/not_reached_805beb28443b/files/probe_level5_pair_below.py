"""Probe the verified 8-9 pair directly below the central row."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from search_level5_threaded import (
    CONTROLLED_PAIR,
    LOWER_FINAL_EIGHT,
    PREFIX,
    SUFFIX,
    apply,
    observe,
)


PAIR_BELOW = (
    (3,) * 5
    + (1,) * 2
    + (4,) * 7
    + (3,) * 4
    + (2,) * 2
    + (4,) * 3
)


def summary(env):
    observation = observe(env)
    return (
        env.levels_completed,
        observation[2],
        observation[3],
        observation[4],
        observation[1],
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(env, PREFIX + CONTROLLED_PAIR + LOWER_FINAL_EIGHT + SUFFIX + PAIR_BELOW)
    print("ROOT", summary(env))
    for shift in range(4):
        branch = env.clone()
        try:
            apply(branch, (4,) * shift)
            print("SHIFT", shift, summary(branch))
            for lift in range(1, 4):
                branch.step(1)
                print("LIFT", shift, lift, summary(branch))
        except IndexError:
            print("INVALID", shift)
    for retraction in range(1, 7):
        branch = env.clone()
        try:
            apply(branch, (3,) * retraction + (1,))
            print("RETRACT_LIFT", retraction, summary(branch))
        except IndexError:
            print("RETRACT_INVALID", retraction)


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
