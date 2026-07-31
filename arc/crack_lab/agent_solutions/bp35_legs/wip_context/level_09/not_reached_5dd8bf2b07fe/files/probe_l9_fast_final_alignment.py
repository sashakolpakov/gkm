"""Flip the compressed boosted route's last switch from each supported column."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_fast_boost_endgame import pre_final, report
from probe_l9_route_deletions import enter_level_9


def run(root, name, prefix, suffix):
    child = pre_final(root)
    for action in prefix:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    report((name, "ALIGNED"), child)
    child.step(*controls(child)[0])
    report((name, "FLIP"), child)
    for index, action in enumerate(suffix, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "C3": (
            ((6, 21, 27), 3),
            (3, 4, (6, 21, 33), (6, 21, 45), 4, 4),
        ),
        "C2": (
            ((6, 21, 27), 3, (6, 15, 27), 3),
            ((6, 15, 33), 4, 4, 4, (6, 21, 27), 4),
        ),
        "C2_REMOVE_Y": (
            ((6, 21, 27), 3, (6, 15, 27), 3, (6, 15, 33)),
            (4, 4, 4, 3, 3),
        ),
    }
    for name, (prefix, suffix) in variants.items():
        run(env, name, prefix, suffix)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
