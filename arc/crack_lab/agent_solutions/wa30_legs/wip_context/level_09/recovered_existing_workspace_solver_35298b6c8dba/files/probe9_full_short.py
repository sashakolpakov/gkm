"""Compare compact level-9 prefixes with the full target/cargo accounting."""

import gkm_try

from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_right_depot import full_target_state
from probe9_verify import boxes


def compact(env):
    empty, filled, occupied = full_target_state(env.frame())
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "cargo": boxes(env.frame(), 4),
        "empty": empty,
        "filled": filled,
        "occupied": occupied,
    }


def run(start, label, route, limit=47):
    clone = start.clone()
    base_level = clone.levels_completed
    for action in route:
        if clone.terminal() or clone.levels_completed > base_level:
            break
        clone.step(action)
    while (
        len(route) < limit
        and not clone.terminal()
        and clone.levels_completed == base_level
    ):
        clone.step(5)
        route = route + [5]
    print("FULL_SHORT", label, len(route), compact(clone), flush=True)


def inspect(env):
    reach_level_9(env)
    short = direct_short_prefix()
    candidates = {
        "idle": short,
        "dismiss": short + [3] * 5 + [5],
        "dismiss_pick": short + [3] * 5 + [5, 3, 3, 1, 5],
        "combined": short + [2, 2, 3, 5, 3, 3, 3],
        "combined_use": short + [2, 2, 3, 5, 3, 3, 3, 5],
    }
    for label, route in candidates.items():
        run(env, label, list(route))


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
