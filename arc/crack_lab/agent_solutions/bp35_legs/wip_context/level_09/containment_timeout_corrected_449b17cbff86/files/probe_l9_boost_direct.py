"""Use the boosted state's lowest switch before it scrolls an upper switch away."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


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


def direct(root):
    child = boosted(root)
    child.step(*controls(child)[0])
    report("BOOST_FLIP", child)
    child.step(*controls(child)[-1])
    report("LOW_FLIP", child)
    child.step(6, 21, 39)
    child.step(3)
    child.step(6, 15, 39)
    child.step(3)
    report("CLEAR_LEFT2", child)
    return child


def run(root, switch_index, actions):
    child = direct(root)
    visible = controls(child)
    child.step(*visible[switch_index])
    report((switch_index, 0, visible[switch_index]), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((switch_index, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for switch_index in range(2):
        run(
            env,
            switch_index,
            ((6, 15, 33), 4, 4, 4, 4, (6, 27, 33)),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
