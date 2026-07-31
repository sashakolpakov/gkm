"""Verify deleting the apparent extra interact after middle-right settles."""

import gkm_try

from probe9_best_mutations import POSITION
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


SHORT_SUFFIX = [2, 4, 5, 1, 3, 2, 2, 5, 1]


def inspect(env):
    reach_level_9(env)
    route = (
        direct_second_prefix()
        + COMBINED_DISMISS_PICK
        + [5]
        + POSITION
        + SHORT_SUFFIX
        + [5]
    )
    state = env.clone()
    base_level = state.levels_completed
    prior = target_state(state.frame())
    for turn, action in enumerate(route, 1):
        state.step(action)
        current = target_state(state.frame())
        if current != prior or turn >= 63:
            print(
                "BEST_SHORT_TRACE",
                turn,
                action,
                state.levels_completed - base_level,
                state.terminal(),
                boxes(state.frame(), 14),
                current,
                flush=True,
            )
        prior = current
        if state.terminal() or state.levels_completed > base_level:
            break
    print(
        "BEST_SHORT_RESULT",
        len(route),
        state.levels_completed - base_level,
        state.terminal(),
        target_state(state.frame()),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
