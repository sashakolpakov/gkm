"""Allow early interactions from the carried-block state above the target."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_position_beam import search
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


TO_TURN_55 = (
    direct_second_prefix()
    + COMBINED_DISMISS_PICK
    + [5, 4]
    + [1] * 6
)


def inspect(env):
    reach_level_9(env)
    state = env.clone()
    for action in TO_TURN_55:
        state.step(action)
    print(
        "TURN55_BEAM_START",
        len(TO_TURN_55),
        boxes(state.frame(), 14),
        boxes(state.frame(), 12),
        target_state(state.frame()),
        flush=True,
    )
    print(
        "TURN55_BEAM_RESULT",
        search(
            [(state, [])],
            max_depth=14,
            beam_width=400,
            max_transitions=25000,
        ),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
