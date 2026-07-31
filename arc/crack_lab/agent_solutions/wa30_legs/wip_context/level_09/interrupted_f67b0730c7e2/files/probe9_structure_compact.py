"""Compact phase maps for the actual level-9 handoff and staged route."""

import gkm_try

from probe9_actual_candidates import state
from probe9_stage_endings import STAGE_BASE
from probe9_stage_finish_beam import ENDING
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import boxes, tile_map


def show(env, turn, label):
    print("STRUCTURE", label, turn, state(env), flush=True)
    print(*tile_map(env.frame()), sep="\n", flush=True)


def inspect(env):
    gkm_try.resumed_solve(env)
    clone = env.clone()
    show(clone, 13, "handoff")

    phase_lengths = (11, 17, 26, 31)
    route = TWO_STAGED + DISMISS
    for turn, action in enumerate(route, 14):
        clone.step(action)
        if turn - 13 in phase_lengths:
            show(clone, turn, "two_stage")

    idle = clone.clone()
    prior = state(idle)
    for turn in range(45, 71):
        idle.step(5)
        current = state(idle)
        if (
            current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["cargo"] != prior["cargo"]
        ):
            print("STRUCTURE_IDLE", turn, current, flush=True)
        prior = current
        if idle.terminal():
            break

    for turn, action in enumerate(STAGE_BASE, 45):
        clone.step(action)
        print("STAGE_STEP", turn, action, boxes(clone.frame(), 0),
              state(clone), flush=True)
    for turn, action in enumerate(ENDING, 45 + len(STAGE_BASE)):
        clone.step(action)
        print("STAGE_STEP", turn, action, boxes(clone.frame(), 0),
              state(clone), flush=True)
    show(clone, 44 + len(STAGE_BASE) + len(ENDING), "local_stage")


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
