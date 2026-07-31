"""Inspect and finish the early-dismissal, lower-staging level-8 branch."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_idle import ROUTE
from probe8_reverse_stage import STAGE_LOWER, compact
from probe8_trace import target_state
from probe9_verify import tile_map


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    clone = env.clone()
    prefix = ROUTE + STAGE_LOWER
    for action in prefix:
        clone.step(action)
    print("FAST8_START", compact(clone, len(prefix)), flush=True)
    print(*tile_map(clone.frame()), sep="\n", flush=True)

    second_stage = [3] * 6 + [1, 5] + [4] * 5 + [5, 2]
    turn = len(prefix)
    prior = target_state(clone.frame())
    for action in second_stage:
        clone.step(action)
        turn += 1
        current = target_state(clone.frame())
        if action == 5 or current != prior:
            print("FAST8_SECOND", compact(clone, turn), flush=True)
        prior = current
    while not clone.terminal() and clone.levels_completed == env.levels_completed:
        clone.step(5)
        turn += 1
        current = target_state(clone.frame())
        if current != prior or turn >= 84:
            print("FAST8_EVENT", compact(clone, turn), flush=True)
        prior = current
    print("FAST8_RESULT", compact(clone, turn), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
