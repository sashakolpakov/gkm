"""Relocate inside the lower maze before the first deep gravity flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def avatar_col(env):
    blobs = [
        blob
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    ]
    return None if not blobs else round((blobs[0].centroid[1] - 3) / 6)


def before_flip(root):
    return replay(root, route()[:110], skips={i for i in SKIPS if i < 110})


def probe(env):
    enter_level_9(env)
    root = before_flip(env)
    print("ROOT", compact(root), "col", avatar_col(root), "controls", controls(root))
    approaches = {
        9: (),
        8: (3,),
        7: (3, 3),
        6: (3, 3, (6, 39, 27), 3),
    }
    for target, approach in approaches.items():
        child = root.clone()
        for action in approach:
            child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            "LANE",
            target,
            "approach",
            approach,
            "terminal",
            bool(child.terminal()),
            "col",
            avatar_col(child),
            "controls",
            controls(child),
            "state",
            compact(child),
        )
        visible = controls(child)
        if not visible or child.terminal():
            continue
        child.step(*visible[0])
        print(
            "FLIP",
            target,
            "terminal",
            bool(child.terminal()),
            "col",
            avatar_col(child),
            "controls",
            controls(child),
            "state",
            compact(child),
        )
        for lefts in range(1, target + 1):
            child.step(3)
            visible = controls(child)
            if visible or child.terminal() or lefts == target:
                print(
                    "RETURN",
                    target,
                    "lefts",
                    lefts,
                    "terminal",
                    bool(child.terminal()),
                    "col",
                    avatar_col(child),
                    "controls",
                    visible,
                    "state",
                    compact(child),
                )
                if visible or child.terminal():
                    break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
