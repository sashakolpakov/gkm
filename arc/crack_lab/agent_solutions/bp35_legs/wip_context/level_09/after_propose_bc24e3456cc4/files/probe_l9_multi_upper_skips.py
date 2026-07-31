"""Screen how many identical opening climb clicks can be omitted."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def avatars(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    actions = route()
    for count in range(11):
        omitted = set(range(11, 11 + count))
        child = replay(env, actions, skips=SKIPS | omitted)
        print(
            "SKIP_COUNT",
            count,
            "applied",
            len(actions) - len(SKIPS | omitted),
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "avatars",
            avatars(child),
            "controls",
            controls(child),
            "grid",
            compact(child)["grid9"],
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
