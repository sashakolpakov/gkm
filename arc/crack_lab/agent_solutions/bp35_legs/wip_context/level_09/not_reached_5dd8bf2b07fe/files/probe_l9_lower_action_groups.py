"""Group every coordinate action by its physical effect at the maze landing."""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS
from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def enter_lower_right(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(9):
        child.step(4)
    return child


def signature(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return bool(env.terminal()), int(env.levels_completed), frame.tobytes()


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14), min_area=2
        )
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    root = enter_lower_right(env)
    print("ROOT", compact(root), "objects", objects(root))
    groups = defaultdict(list)
    reps = {}
    actions = [3, 4, 7]
    actions.extend(
        (6, x, y)
        for y in ROW_ANCHORS
        for x in COL_ANCHORS
    )
    for action in actions:
        child = root.clone()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        key = signature(child)
        groups[key].append(action)
        reps[key] = child
    for key, members in sorted(
        groups.items(), key=lambda item: (len(item[1]), repr(item[1]))
    ):
        child = reps[key]
        print(
            "GROUP",
            members,
            "terminal",
            bool(child.terminal()),
            "state",
            compact(child),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
