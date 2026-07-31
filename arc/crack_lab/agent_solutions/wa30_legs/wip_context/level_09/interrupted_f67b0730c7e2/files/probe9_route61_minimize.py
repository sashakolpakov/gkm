"""Greedily delete actions from the verified 61-turn level-9 route."""

import os

import gkm_try

from probe9_minimize_win import ROUTE
from probe9_prefix_shortcuts import reach_level_9
from probe_minimize_segments import minimize
from probe_pair_minimize import encode, pair_minimize, triple_minimize


def inspect(env):
    reach_level_9(env)
    mode = os.environ.get("GKM_L9_MIN_MODE", "segment")
    minimizer = {
        "segment": minimize,
        "pair": pair_minimize,
        "triple": triple_minimize,
    }[mode]
    best, turns = minimizer(env, ROUTE)
    print(
        "L9_ROUTE61_MIN_RESULT",
        mode,
        len(best),
        turns,
        encode(best),
        best,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
