"""Test nine-move positioning while reserving settle and reward ticks."""

from itertools import product

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_short_position_orders import (
    positioned_states,
    repair_suffixes,
    replay,
)
from probe9_verify import boxes, target_state


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    pickup_prefix = direct_second_prefix() + COMBINED_DISMISS_PICK + [5]
    for action in pickup_prefix:
        picked.step(action)
    states = {}
    for up, right in ((6, 3), (5, 4), (4, 5), (3, 6)):
        for state, path in positioned_states(picked, up, right):
            states.setdefault(
                arr(state.frame()).tobytes(),
                (state, path, (up, right)),
            )
    print("POSITION_NINE_STATES", len(states), flush=True)
    best = None
    for state, path, counts in states.values():
        for suffix in repair_suffixes():
            if len(suffix) > 10:
                continue
            child = replay(state, suffix)
            target = target_state(child.frame())
            score = (
                child.levels_completed - state.levels_completed,
                len(target["filled"]),
                -len(target["empty"]),
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    counts,
                    path,
                    suffix,
                    boxes(child.frame(), 14),
                    target,
                )
            if child.levels_completed > state.levels_completed:
                print("POSITION_NINE_WIN", counts, path, suffix, flush=True)
                return
            remaining = 69 - (48 + 9 + len(suffix))
            if (
                remaining <= 0
                or remaining > 3
                or (len(target["filled"]) < 7 and target["empty"])
            ):
                continue
            for tail in product(child.actions, repeat=remaining):
                rewarded = replay(child, tail)
                if rewarded.levels_completed > state.levels_completed:
                    print(
                        "POSITION_NINE_WIN",
                        counts,
                        path,
                        suffix + list(tail),
                        flush=True,
                    )
                    return
    print("POSITION_NINE_BEST", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
