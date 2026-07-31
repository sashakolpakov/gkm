"""Clear the outer neighbor before entering either height-six endpoint cell."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_presecond_upper_height6_exits import height6, state
from probe_l9_route_deletions import enter_level_9


def apply(child, name, actions):
    for index, action in enumerate(actions, 1):
        child.step(*action)
        state((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    wait = height6(env)
    apply(wait, "WAIT", ((6, 3, 27),) * 8)
    right = height6(env)
    apply(
        right,
        "RIGHT",
        (
            (6, 39, 39), (4,),
            (6, 45, 39), (4,),
            (6, 51, 39),
            (6, 57, 39),
            (4,),
            (4,),
        ),
    )
    left = height6(env)
    apply(
        left,
        "LEFT",
        (
            (6, 27, 39), (3,),
            (6, 21, 39), (3,),
            (6, 15, 39),
            (6, 9, 39),
            (3,),
            (3,),
        ),
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
