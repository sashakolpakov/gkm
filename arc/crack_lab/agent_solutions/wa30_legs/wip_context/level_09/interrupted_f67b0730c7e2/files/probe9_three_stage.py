"""Test feeding a third local cargo after the two remote level-9 stages."""

import gkm_try

from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import boxes, target_state, tile_map


STAGE_LOCAL = (
    [3] * 2 + [5]
    + [4] + [1] * 5 + [5, 2]
)


def compact(env, turn):
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "cargo": boxes(env.frame(), 4),
        "target": target_state(env.frame()),
    }


def inspect(env):
    gkm_try.resumed_solve(env)
    clone = env.clone()
    for action in TWO_STAGED + DISMISS:
        clone.step(action)
    for action in STAGE_LOCAL:
        clone.step(action)
    turn = 13 + len(TWO_STAGED + DISMISS + STAGE_LOCAL)
    print("THREE_STAGE", compact(clone, turn), flush=True)
    print("THREE_STAGE_MAP", *tile_map(clone.frame()), sep="\n", flush=True)

    prior = target_state(clone.frame())
    while not clone.terminal() and clone.levels_completed == env.levels_completed:
        clone.step(5)
        turn += 1
        current = target_state(clone.frame())
        if current != prior or turn >= 66:
            print("THREE_EVENT", compact(clone, turn), flush=True)
        prior = current


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
