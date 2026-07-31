"""Descend safely after opening and entering the first lower-maze catch gate."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def enter_lower_right(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(9):
        child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = enter_lower_right(env)
    actions = [3, 3, (6, 39, 27), 3]
    actions.extend([(6, 39, 33)] * 8)
    print("START", compact(child), "goals", goals(child))
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            "STEP",
            index,
            action,
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "goals",
            goals(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
