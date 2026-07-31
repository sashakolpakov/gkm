"""Trace the forced alternating-control zigzag from the one-skip endgame."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip_upper_endgame import objects, root_skip


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "goals",
        goals(env),
        "state",
        compact(env),
        "objects",
        objects(env),
    )


def run(root, name, actions):
    child = root_skip(root)
    report((name, "START"), child)
    for index, action in enumerate(actions, 1):
        if action == "top":
            visible = controls(child)
            action = visible[0]
        elif action == "bottom":
            visible = controls(child)
            action = visible[-1]
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    run(env, "FORCED", ("top", 4, "bottom", 3))
    run(env, "NO_SECOND_FLIP", ("top", 4, 3, 3, 3))
    run(env, "BOTTOM_FIRST", ("bottom", 4, "top", 3))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
