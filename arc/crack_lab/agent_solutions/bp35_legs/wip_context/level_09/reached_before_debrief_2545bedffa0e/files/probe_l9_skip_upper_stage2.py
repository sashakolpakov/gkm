"""Explore the higher c4 landing reached through the one-skip corridor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_shift
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip_upper_next_room import stage, step


def stage2(root):
    child = stage(root)
    step(child, controls(child)[0])
    step(child, 4)
    step(child, 4)
    return child


def brief(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        tuple(controls(env)),
        compact(env)["grid9"],
    )


def run(root, name, actions):
    child = stage2(root)
    print(name, 0, brief(child), flush=True)
    for index, token in enumerate(actions, 1):
        action = token
        if token == "switch":
            action = controls(child)[-1]
        before = child.frame()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            name,
            index,
            action,
            "gain",
            band_shift(before, child.frame()),
            brief(child),
            flush=True,
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "LEFT": (3, 3, 3, 3),
        "SWITCH_LEFT": ("switch", 3, 3, 3, 3),
        "CLEAR_LEFT": ((6, 15, 27), 3, 3, 3),
        "SWITCH_CLEAR_LEFT": ("switch", (6, 15, 27), 3, 3, 3),
        "DROP_C4": ((6, 27, 33),) * 5,
        "SWITCH_RISE_C4": ("switch", (6, 27, 21), (6, 27, 21), 3),
        "DROP_C3": (3, (6, 21, 33), (6, 21, 33), 3),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
