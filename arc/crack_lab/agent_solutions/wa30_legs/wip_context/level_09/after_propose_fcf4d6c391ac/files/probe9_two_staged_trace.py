"""Trace the compact two-stage level-9 route at meaningful events."""

import gkm_try

from probe9_actual_candidates import state
from probe9_verify import boxes, target_state, tile_map


TWO_STAGED = (
    [3, 1, 5] + [3] * 6 + [1, 5]
    + [4] * 4 + [1, 5]
    + [2] + [3] * 6 + [1, 5]
)
DISMISS = [1, 3, 3, 3, 5]
LOCAL_FINISH = (
    [3] * 2 + [5, 4]
    + [1] * 6 + [4] * 5
    + [5, 1, 3, 2, 5, 2, 5, 1]
)


def compact(env, turn, action):
    return {
        "turn": turn,
        "action": action,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "target": target_state(env.frame()),
    }


def inspect(env):
    gkm_try.resumed_solve(env)
    clone = env.clone()
    prior = target_state(clone.frame())
    route = TWO_STAGED + DISMISS + LOCAL_FINISH
    for offset, action in enumerate(route, 1):
        clone.step(action)
        turn = 13 + offset
        current = target_state(clone.frame())
        if current != prior or action == 5 or turn >= 62:
            print("TWO_TRACE", compact(clone, turn, action), flush=True)
        if turn == 44:
            print("TWO_DISMISS_MAP", *tile_map(clone.frame()), sep="\n",
                  flush=True)
        prior = current
        if clone.terminal() or clone.levels_completed > env.levels_completed:
            break
    print("TWO_FINAL", state(clone), flush=True)
    print("TWO_MAP", *tile_map(clone.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
