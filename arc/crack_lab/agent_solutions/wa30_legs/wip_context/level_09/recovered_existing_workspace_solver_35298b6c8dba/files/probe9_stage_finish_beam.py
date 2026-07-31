"""Finish from the turn-56 carry state that induces middle-right delivery."""

import gkm_try

from probe9_stage_endings import STAGE_BASE
from probe9_two_stage_beam import search
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import boxes, target_state


ENDING = [1, 4, 1]


def inspect(env):
    gkm_try.resumed_solve(env)
    state = env.clone()
    prefix = TWO_STAGED + DISMISS + STAGE_BASE + ENDING
    for action in prefix:
        state.step(action)
    print(
        "STAGE_FINISH_START",
        13 + len(prefix),
        boxes(state.frame(), 14),
        boxes(state.frame(), 12),
        boxes(state.frame(), 4),
        target_state(state.frame()),
        flush=True,
    )
    print(
        "STAGE_FINISH_RESULT",
        search(state, max_depth=13, beam_width=400, max_transitions=25000),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
