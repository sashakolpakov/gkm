"""Open column four, enter it, and remove its yellow support."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_col5_yellow_trapdoor import col5_depth4
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14), min_area=2
        )
        if blob.bbox[0] < 63
    )


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
    run(
        env,
        "OPEN_ENTER_REMOVE",
        [(6, 27, 27), 3, (6, 27, 35), (6, 27, 35), 3, 4],
    )
    run(
        env,
        "REMOVE_OPEN_ENTER",
        [(6, 27, 35), (6, 27, 27), 3, (6, 27, 35), 3, 4],
    )
    run(
        env,
        "REMOVE_OPEN_DESCEND",
        [(6, 27, 35), (6, 27, 27), 3, *([(6, 27, 35)] * 8)],
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
