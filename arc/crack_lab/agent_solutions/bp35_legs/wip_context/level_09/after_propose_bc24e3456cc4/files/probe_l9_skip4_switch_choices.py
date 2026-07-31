"""Carry each pre-corridor switch through the four-skip goal shortcut."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_twelve_fast_frontier import SKIPS


OMITTED = SKIPS | set(range(11, 15))


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "controls",
        controls(env),
        "goals",
        boxes(env, 7),
        "avatar",
        boxes(env, 9),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def before_flip(root):
    child = root.clone()
    for index, (_, action) in enumerate(route()):
        if index > 26:
            break
        if index not in OMITTED:
            step(child, action)
    return child


def run(root, switch_index):
    child = before_flip(root)
    visible = controls(child)
    report((switch_index, "PRE"), child)
    child.step(*visible[switch_index])
    report((switch_index, "FLIP"), child)
    for index in range(28, 34):
        action = route()[index][1]
        step(child, action)
        report((switch_index, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    visible = controls(child)
    if visible:
        child.step(*visible[-1])
        report((switch_index, "WALL_FLIP"), child)
        for action in (3, 3, 4, 4, (6, 15, 33), (6, 15, 45)):
            step(child, action)
            report((switch_index, action), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                return


def probe(env):
    enter_level_9(env)
    for switch_index in range(3):
        run(env, switch_index)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
