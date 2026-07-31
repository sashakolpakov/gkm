"""Take the lateral c8-to-c2 exit after seven safe exterior c9 climbs."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, relevant_full
from probe_l9_presecond_c9_descent import enter_c9
from probe_l9_route_deletions import enter_level_9


def top_c9(root):
    child = enter_c9(root)
    child.step(*controls(child)[0])
    for _ in range(7):
        child.step(6, 57, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = top_c9(env)
    report("TOP_C9", child)
    actions = []
    for x in (51, 45, 39, 33, 27, 21, 15):
        actions.extend(((6, x, 39), 3))
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((index, action), child)
        print(
            "STATE",
            index,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "full",
            relevant_full(child),
            flush=True,
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
