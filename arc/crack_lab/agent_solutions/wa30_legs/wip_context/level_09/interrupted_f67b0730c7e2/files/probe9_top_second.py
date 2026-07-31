"""Test placing the second remote level-9 cargo into top-right first."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state, tile_map


PLACE_TOP_RIGHT = [3] * 5 + [1] * 5 + [3, 5, 4]


def compact(env, turn):
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "cargo": boxes(env.frame(), 4),
        "target": target_state(env.frame()),
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    clone = env.clone()
    route = direct_second_prefix()[:29] + PLACE_TOP_RIGHT
    for action in route:
        clone.step(action)
    print("TOP_SECOND", compact(clone, len(route)), flush=True)
    print("TOP_SECOND_MAP", *tile_map(clone.frame()), sep="\n", flush=True)

    turn = len(route)
    prior = target_state(clone.frame())
    while turn < 70 and not clone.terminal():
        clone.step(5)
        turn += 1
        current = target_state(clone.frame())
        if current != prior or turn % 4 == 0:
            print("TOP_SECOND_IDLE", compact(clone, turn), flush=True)
        prior = current


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
