"""Trace target events in the exact all-eight-at-69 level-9 route."""

import gkm_try

from probe9_best_mutations import POSITION, SUFFIX
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


def inspect(env):
    reach_level_9(env)
    route = (
        direct_second_prefix()
        + COMBINED_DISMISS_PICK
        + [5]
        + POSITION
        + SUFFIX
    )
    state = env.clone()
    prior = target_state(state.frame())
    for turn, action in enumerate(route, 1):
        state.step(action)
        current = target_state(state.frame())
        if current != prior or turn >= 56:
            print(
                "BEST_TRACE",
                {
                    "turn": turn,
                    "action": action,
                    "level": state.levels_completed,
                    "terminal": state.terminal(),
                    "avatar": boxes(state.frame(), 14),
                    "helpers": boxes(state.frame(), 12),
                    "empty": current["empty"],
                    "filled": current["filled"],
                    "signatures": current["signatures"],
                },
                flush=True,
            )
        prior = current


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
