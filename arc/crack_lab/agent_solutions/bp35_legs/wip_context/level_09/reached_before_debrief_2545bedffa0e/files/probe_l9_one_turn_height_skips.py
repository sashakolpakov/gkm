"""Screen one representative from each repeated height-changing prefix block."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9, replay, route, summary
from probe_l9_twelve_fast_frontier import SKIPS


CANDIDATES = (0, 4, 11, 20, 28, 29, 35, 41, 43, 47, 49, 50, 63, 73, 75, 83, 84, 103, 109, 110, 111)


def probe(env):
    enter_level_9(env)
    actions = route()
    for index in CANDIDATES:
        child = replay(env, actions, skips=SKIPS | {index})
        result = summary(child)
        print(
            "SKIP",
            index,
            actions[index],
            "terminal",
            result["terminal"],
            "level",
            result["level"],
            "avatar",
            result["avatar"],
            "controls",
            len(controls(child)),
            "grid",
            result["grid"],
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
