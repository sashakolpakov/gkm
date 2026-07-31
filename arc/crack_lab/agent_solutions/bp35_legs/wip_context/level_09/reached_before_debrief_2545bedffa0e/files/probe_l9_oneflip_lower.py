"""Carry the three-skip switch toward the prize wall without consuming it."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_multiskip_drop_flip import dropped
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


def oneflip(root, lefts):
    child = dropped(root, 1)
    child.step(*controls(child)[-1])
    for _ in range(lefts):
        child.step(3)
    return child


def run(root, lefts, drop_x, actions):
    child = oneflip(root, lefts)
    report((lefts, drop_x, 0), child)
    child.step(6, drop_x, 33)
    report((lefts, drop_x, 1, "DROP"), child)
    for index, token in enumerate(actions, 2):
        action = token
        if token == "control":
            visible = controls(child)
            if not visible:
                report((lefts, drop_x, index, "NO_CONTROL"), child)
                break
            action = visible[-1]
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((lefts, drop_x, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for lefts, drop_x in ((0, 27), (1, 21), (2, 15)):
        run(env, lefts, drop_x, (4, 4, 4, "control", 3, 3, 4, 4))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
