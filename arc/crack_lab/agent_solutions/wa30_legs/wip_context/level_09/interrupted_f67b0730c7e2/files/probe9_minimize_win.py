"""Minimize the verified 61-turn full level-9 route."""

import gkm_try

from probe9_prefix_shortcuts import reach_level_9
from probe_pair_minimize import encode, pair_minimize


ROUTE = (
    [2] + [4] * 6 + [1, 5] + [1] * 2 + [5, 2]
    + [3] * 2 + [1, 5] + [4, 1, 5, 2]
    + [3] * 2 + [1] * 3 + [5] + [1] * 2 + [5, 2]
    + [2] * 3 + [3] * 5 + [2] * 2 + [3, 5]
    + [3] * 4 + [1, 5] + [4] * 7 + [1] * 3 + [3, 5]
)


def inspect(env):
    reach_level_9(env)
    print("L9_MIN_BASE", len(ROUTE), encode(ROUTE), flush=True)
    best, turns = pair_minimize(env, ROUTE)
    print(
        "L9_MIN_RESULT",
        len(best),
        turns,
        encode(best),
        best,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
