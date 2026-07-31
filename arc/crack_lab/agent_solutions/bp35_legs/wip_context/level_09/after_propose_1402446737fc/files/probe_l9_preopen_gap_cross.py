"""Cross the pre-opened yellow gap using the verified arrest catch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_preopen_yellow_flip import yellows
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_final_alignment import aligned


def root_state(root):
    child = aligned(root, 5, 5)
    for _ in range(2):
        action = max(yellows(child), key=lambda item: item[1])
        child.step(*action)
    child.step(*controls(child)[-1])
    child.step(6, 21, 27)
    return child


def probe(env):
    enter_level_9(env)
    child = root_state(env)
    report("ROOT", child)
    actions = (3, (6, 21, 39), 3, (6, 15, 39), 3)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
