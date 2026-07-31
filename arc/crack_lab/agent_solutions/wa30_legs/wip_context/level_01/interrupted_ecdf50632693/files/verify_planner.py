"""One-shot clone verification for the reusable carry leg."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import plan_rimmed_blocks_to_targets


def verify(env):
    path = plan_rimmed_blocks_to_targets(env.frame())
    clone = env.clone()
    base_level = int(clone.levels_completed)
    for action in path or ():
        if clone.levels_completed > base_level:
            break
        clone.step(action)
    print(
        "PLAN",
        path,
        "LENGTH",
        len(path or ()),
        "LEVEL",
        clone.levels_completed,
    )


arena.run_program("wa30", verify)
