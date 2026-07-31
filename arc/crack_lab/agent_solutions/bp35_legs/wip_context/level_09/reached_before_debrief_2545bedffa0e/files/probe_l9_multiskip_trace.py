"""Trace when the prize chamber becomes separated in the six-skip replay."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_twelve_fast_frontier import SKIPS


def boxes(env, colors):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=colors, min_area=2)
        if blob.bbox[0] < 63
    )


def summary(env):
    return {
        "terminal": bool(env.terminal()),
        "level": int(env.levels_completed) + 1,
        "avatar": boxes(env, (9, 11)),
        "goal": boxes(env, (7,)),
        "controls": controls(env),
        "grid": compact(env)["grid9"],
    }


def trace(root, count):
    child = root.clone()
    skips = SKIPS | set(range(11, 11 + count))
    applied = 0
    for index, (section, action) in enumerate(route()):
        if index in skips:
            continue
        step(child, action)
        applied += 1
        if index >= 75:
            print(
                count,
                index,
                applied,
                section,
                action,
                summary(child),
                flush=True,
            )
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    trace(env, 6)
    for count in (6, 10):
        child = env.clone()
        skips = SKIPS | set(range(11, 11 + count))
        for index, (_, action) in enumerate(route()):
            if index not in skips:
                step(child, action)
        print("GOAL_ROOT", count, summary(child), flush=True)
        for action in ((6, 15, 46), 7):
            branch = child.clone()
            step(branch, action)
            print("GOAL_ACTION", count, action, summary(branch), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
