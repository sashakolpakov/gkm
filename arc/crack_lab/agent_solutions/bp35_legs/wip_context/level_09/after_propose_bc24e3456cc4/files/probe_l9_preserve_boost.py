"""Select the useful switch after preserving the gate and revealing a fourth."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


UP = (6, 27, 33)


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "controls",
        controls(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def boosted(root):
    child = root_for(root, 3)
    for action in ((6, 21, 39), 4, (6, 27, 39), 4, UP, UP, UP):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def run(root, switch_index, actions):
    child = boosted(root)
    visible = controls(child)
    report((switch_index, 0), child)
    child.step(*visible[switch_index])
    report((switch_index, 1, visible[switch_index]), child)
    for index, action in enumerate(actions, 2):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((switch_index, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for switch_index in range(4):
        run(env, switch_index, (3, 4, (6, 27, 45), (6, 27, 45)))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
