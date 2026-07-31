"""Open the first catch gate in the lower maze and seek its wall gap."""

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
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=3
        )
        if blob.bbox[0] < 63 and (blob.color != 15 or blob.area == 21)
    )


def enter_lower_right(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(9):
        child.step(4)
    return child


def run(root, name, actions):
    child = enter_lower_right(root)
    print(name, "START", compact(child), "objects", objects(child))
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            name,
            index,
            action,
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "objects",
            objects(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    run(env, "LEFT", [3, 3, 3, 3])
    run(
        env,
        "OPEN_CLIMB",
        [3, 3, (6, 39, 27), 3, (6, 39, 21), (6, 39, 21), 3, 3],
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
