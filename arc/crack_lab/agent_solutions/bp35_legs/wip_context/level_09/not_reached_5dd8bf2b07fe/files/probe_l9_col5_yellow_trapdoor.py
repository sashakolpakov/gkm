"""Use the yellow block beside the direct column-five descent as a trapdoor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14), min_area=2
        )
        if blob.bbox[0] < 63
    )


def col5_depth4(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(5):
        child.step(4)
    for _ in range(4):
        child.step(6, 33, 35)
    return child


def run(root, name, actions):
    child = col5_depth4(root)
    print(name, "START", compact(child), "objects", objects(child))
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            name,
            index,
            action,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "controls",
            controls(child),
            "state",
            compact(child),
            "objects",
            objects(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    run(env, "REMOVE_THEN_LEFT", [(6, 27, 35), 3, 3, (6, 21, 35)])
    run(env, "LEFT_THEN_REMOVE", [3, (6, 27, 35), 3, (6, 21, 35)])


if __name__ == "__main__":
    arena.run_program("bp35", probe)
