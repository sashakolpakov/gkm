"""Compare the three gravity switches at the staged four-skip frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_search import supported


def run(root, switch_index, actions):
    child = supported(root)
    visible = controls(child)
    child.step(*visible[switch_index])
    report((switch_index, visible[switch_index], 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((switch_index, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for switch_index in range(3):
        run(
            env,
            switch_index,
            ((6, 21, 27), 3, (6, 15, 27), 3, 4, 4),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
