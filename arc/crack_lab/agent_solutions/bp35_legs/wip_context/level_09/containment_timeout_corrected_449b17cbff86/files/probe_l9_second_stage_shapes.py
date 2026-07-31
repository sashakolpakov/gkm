"""Shape one staged shaft before the second flip to remove its entry gate."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def avatars(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    chamber = replay(env, route(), skips=SKIPS)
    x = 21
    variants = {
        "BASE": ((6, x, 51),),
        "UP": ((6, x, 51), (6, x, 45)),
        "DOWN": ((6, x, 51), (6, x, 57)),
        "UP_BACK": ((6, x, 51), (6, x, 45), (6, x, 39)),
        "DOWN_BACK": ((6, x, 51), (6, x, 57), (6, x, 51)),
    }
    for name, actions in variants.items():
        child = chamber.clone()
        for action in actions:
            child.step(*action)
        before = compact(child)
        child.step(*controls(child)[0])
        flipped = compact(child)
        for _ in range(3):
            child.step(4)
        print(
            name,
            "actions",
            actions,
            "before",
            before,
            "flipped",
            flipped,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatars(child),
            "walked",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
