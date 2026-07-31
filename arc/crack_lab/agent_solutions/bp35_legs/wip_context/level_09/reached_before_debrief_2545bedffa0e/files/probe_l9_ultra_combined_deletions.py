"""Verify commuting combinations of exact boosted-prefix deletions."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_faster_prefix_deletions import candidate, physical
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_cycle_stage import EXTRA_SKIPS


ROOM = {28, 29, *range(35, 42), *range(43, 48), 49, 63}
BRIDGE = {75, 76, 77, 78, 79, 82}
RETURN = set(range(111, 120))


def probe(env):
    enter_level_9(env)
    target = candidate(env, EXTRA_SKIPS)
    target_key = physical(target)
    variants = (
        ("ROOM", ROOM),
        ("BRIDGE", BRIDGE),
        ("RETURN", RETURN),
        ("ROOM_BRIDGE", ROOM | BRIDGE),
        ("ROOM_RETURN", ROOM | RETURN),
        ("ALL", ROOM | BRIDGE | RETURN),
    )
    for name, extra in variants:
        child = candidate(env, EXTRA_SKIPS | extra)
        print(
            name,
            "count",
            len(extra),
            "same",
            physical(child) == target_key,
            "terminal",
            bool(child.terminal()),
            "controls",
            len(controls(child)),
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
