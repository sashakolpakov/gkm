"""Check whether a prior gravity/catch leg transfers to pristine level 9."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    cross_persistent_support_rooms,
    cross_staged_gravity_zigzag,
    cross_support_ladder_round_trip,
)
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


NAME = sys.argv[1] if len(sys.argv) > 1 else "ladder"
LEGS = {
    "persistent": cross_persistent_support_rooms,
    "zigzag": cross_staged_gravity_zigzag,
    "ladder": cross_support_ladder_round_trip,
}


def probe(env):
    enter_level_9(env)
    before = int(env.levels_completed)
    result = LEGS[NAME](env)
    print(
        NAME,
        "result",
        result,
        "levels",
        int(env.levels_completed),
        "gain",
        int(env.levels_completed) - before,
        "terminal",
        bool(env.terminal()),
        "controls",
        controls(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
