"""Verify the one-turn-faster column-three entry through the full finish."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals
from probe_l9_staged_entry_trace import replay


def avatar(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "avatar",
        avatar(env),
        "controls",
        controls(env),
        "goals",
        goals(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def run(root, name, entry_skip, actions):
    child = replay(root, skips=(entry_skip,))
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    finish = (
        *((6, 21, 33),) * 5,
        (6, 27, 27),
        4,
        4,
        4,
        (6, 45, 33),
        4,
        (6, 51, 27),
        4,
        (6, 57, 27),
        4,
    )
    run(env, "NO_PRESTAGE", 0, finish)
    run(env, "NO_FINAL_ENTRY_MOVE", 5, finish)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
