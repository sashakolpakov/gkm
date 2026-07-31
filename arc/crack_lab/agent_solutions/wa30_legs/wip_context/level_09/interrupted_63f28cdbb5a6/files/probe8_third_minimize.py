"""Minimize the verified three-transfer level-8 route."""

import gkm_try

from probe8_current_tail import PREFIX
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe_pair_minimize import encode, pair_minimize


THIRD = [1] + [3] * 10 + [1, 5, 2] + [4] * 11 + [1, 5, 2]
ROUTE = PREFIX + THIRD + [5] * 13


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    print("THIRD_MIN_BASE", len(ROUTE), encode(ROUTE), flush=True)
    best, turns = pair_minimize(env, ROUTE)
    print(
        "THIRD_MIN_RESULT",
        len(best),
        turns,
        encode(best),
        best,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
