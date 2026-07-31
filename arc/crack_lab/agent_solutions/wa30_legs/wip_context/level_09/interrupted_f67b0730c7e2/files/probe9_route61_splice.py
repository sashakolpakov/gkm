"""Test arbitrary contiguous phase deletions from the 61-turn route."""

import gkm_try

from probe9_minimize_win import ROUTE
from probe9_prefix_shortcuts import reach_level_9
from probe_minimize_segments import evaluate


def inspect(env):
    reach_level_9(env)
    tested = 0
    wins = []
    for size in range(30, 1, -1):
        for start in range(0, len(ROUTE) - size + 1):
            candidate = ROUTE[:start] + ROUTE[start + size:]
            won, turns = evaluate(env, candidate, 60)
            tested += 1
            if won:
                wins.append((turns, size, start, candidate))
                print("ROUTE61_SPLICE_WIN", wins[-1], flush=True)
        if wins:
            break
    print("ROUTE61_SPLICE_RESULT", tested, wins, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
