"""Combine the two verified deletions and inspect the second-flip window."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route, summary


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_level_9(env)
    chamber = replay(env, route(), skips=(42, 48))
    print("CHAMBER", summary(chamber))
    chamber.step(*controls(chamber)[0])
    print("FLIP", compact(chamber), "avatar", avatar(chamber))
    for step in range(1, 12):
        chamber.step(4)
        print(
            "RIGHT",
            step,
            compact(chamber),
            "terminal",
            bool(chamber.terminal()),
            "avatar",
            avatar(chamber),
            "controls",
            controls(chamber),
        )
        if chamber.terminal() or int(chamber.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
