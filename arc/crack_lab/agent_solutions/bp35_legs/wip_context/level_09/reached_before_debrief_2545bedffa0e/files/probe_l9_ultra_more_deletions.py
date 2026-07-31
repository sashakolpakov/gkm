"""Find one more exact no-op in the complete compressed boosted prefix."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_faster_prefix_deletions import candidate, physical
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_cycle_stage import EXTRA_SKIPS


CANDIDATES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    28, 29,
    35, 36, 37, 38, 39, 40, 41,
    43, 44, 45, 46, 47, 49, 63,
    75, 76, 77, 78, 79, 80, 81, 82,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
)


def probe(env):
    enter_level_9(env)
    target = candidate(env, EXTRA_SKIPS)
    target_key = physical(target)
    for index in CANDIDATES:
        child = candidate(env, EXTRA_SKIPS | {index})
        print(
            index,
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
