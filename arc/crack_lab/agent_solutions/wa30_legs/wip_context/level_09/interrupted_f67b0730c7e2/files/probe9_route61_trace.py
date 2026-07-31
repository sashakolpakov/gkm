"""Trace interaction and target events in the verified 61-turn route."""

import gkm_try

from probe9_minimize_win import ROUTE
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


def compact(env):
    target = target_state(env.frame())
    return {
        "level": env.levels_completed,
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "cargo": boxes(env.frame(), 4),
        "empty": target["empty"],
        "filled": target["filled"],
    }


def inspect(env):
    reach_level_9(env)
    clone = env.clone()
    base_level = clone.levels_completed
    prior = compact(clone)
    print("ROUTE61_EVENT", 0, None, prior, flush=True)
    for turn, action in enumerate(ROUTE, 1):
        clone.step(action)
        current = compact(clone)
        if (
            action == 5
            or current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["thief"] != prior["thief"]
            or current["level"] != prior["level"]
        ):
            print("ROUTE61_EVENT", turn, action, current, flush=True)
        prior = current
        if clone.levels_completed > base_level or clone.terminal():
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
