"""Measure whether stack undo restores the level-nine action horizon."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def flipped(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    return child


def run(root, name, actions):
    child = flipped(root)
    start = compact(child)
    for index, action in enumerate(actions, 1):
        child.step(action)
        if child.terminal() or int(child.levels_completed) >= 9:
            print(name, "terminal_at", index, "state", compact(child))
            return
    print(name, "survived", len(actions), "start", start, "state", compact(child))


def probe(env):
    enter_level_9(env)
    traced = flipped(env)
    print("TRACE", 0, "controls", controls(traced), "state", compact(traced))
    for index in range(1, 20):
        traced.step(7)
        print(
            "TRACE",
            index,
            "terminal",
            bool(traced.terminal()),
            "controls",
            controls(traced),
            "state",
            compact(traced),
        )
    run(env, "undo_only", [7] * 30)
    run(env, "right_undo", [value for _ in range(15) for value in (4, 7)])
    run(env, "right_left_undo", [value for _ in range(10) for value in (4, 3, 7)])


if __name__ == "__main__":
    arena.run_program("bp35", probe)
