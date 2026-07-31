"""Trace and delete actions from the six-turn staged column-three entry."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import chamber, goals


def avatar(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def brief(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        tuple(controls(env)),
        avatar(env),
        goals(env),
        compact(env)["grid9"],
    )


def entry_actions(root):
    child = chamber(root)
    return (
        (6, 21, 51),
        controls(child)[0],
        4,
        4,
        (6, 21, 27),
        4,
    )


def replay(root, skips=()):
    child = chamber(root)
    actions = entry_actions(root)
    for index, action in enumerate(actions):
        if index in skips:
            continue
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    child = chamber(env)
    print("START", brief(child), flush=True)
    for index, action in enumerate(entry_actions(env)):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print("STEP", index, action, brief(child), flush=True)
    for index, action in enumerate(entry_actions(env)):
        child = replay(env, skips=(index,))
        print("SKIP", index, action, brief(child), flush=True)
        if child.terminal():
            continue
        child.step(6, 21, 33)
        print("SKIP_DROP", index, brief(child), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
