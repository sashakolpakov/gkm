"""Try to expose the second control row before the nine-step return walk."""

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
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    )


def before_return(root):
    actions = [item for item in route() if item[0] != "return_left"]
    return replay(root, actions, skips={i for i in SKIPS if i < len(actions)})


def probe(env):
    enter_level_9(env)
    root = before_return(env)
    print("ROOT", compact(root), "controls", controls(root), "objects", objects(root))
    actions = [
        3,
        4,
        7,
        (6, 57, 33),
        (6, 57, 39),
        (6, 57, 45),
        (6, 51, 45),
        (6, 51, 39),
        (6, 3, 41),
    ]
    for action in actions:
        child = root.clone()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            "ACTION",
            action,
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "state",
            compact(child),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
