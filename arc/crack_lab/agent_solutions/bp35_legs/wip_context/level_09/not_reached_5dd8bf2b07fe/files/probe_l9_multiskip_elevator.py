"""Test whether the three-skip surviving switch forms a useful two-flip elevator."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_multiskip_drop_flip import DROP, dropped
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


def elevator(root, depth=1):
    child = dropped(root, depth)
    child.step(*controls(child)[-1])
    child.step(3)
    child.step(3)
    child.step(*controls(child)[-1])
    return child


def run(root, name, actions):
    child = elevator(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        if action == "control":
            visible = controls(child)
            if not visible:
                report((name, index, "NO_CONTROL"), child)
                break
            action = visible[-1]
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "LEFT": (3,) * 10,
        "RIGHT": (4,) * 10,
        "DROP_C2": ((6, 15, 33),) * 8,
        "DROP_C3": ((6, 21, 33),) * 8,
        "DROP_C4": (DROP,) * 8,
        "CLEAR_RIGHT": ((6, 21, 27), 4, (6, 27, 27), 4, 4, 4),
        "CONTROL": ("control", 3, 3, 4, 4),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
