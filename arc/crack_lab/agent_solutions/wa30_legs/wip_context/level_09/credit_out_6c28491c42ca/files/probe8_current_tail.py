"""Inspect actionable cargo near the targets after the current two transfers."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_stage import compact
from probe9_verify import tile_map


PREFIX = (
    [4] * 8 + [2] * 5 + [3] * 3 + [5]
    + [4] * 3 + [1] * 5 + [3] * 5 + [1] * 4
    + [3, 1] + [5] * 3
    + [3, 5, 1] + [4] * 8 + [5, 2]
    + [3] * 10 + [1, 5, 2] + [4] * 11 + [1, 5, 2]
)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    clone = env.clone()
    for turn, action in enumerate(PREFIX, 1):
        clone.step(action)
        if turn == 52:
            print("CURRENT_TAIL", turn, compact(clone, turn), flush=True)
            print(*tile_map(clone.frame()), sep="\n", flush=True)
    print("CURRENT_TAIL", len(PREFIX), compact(clone, len(PREFIX)), flush=True)
    print(*tile_map(clone.frame()), sep="\n", flush=True)

    third = [1] + [3] * 10 + [1, 5, 2] + [4] * 11 + [1, 5, 2]
    candidate = clone.clone()
    base_level = candidate.levels_completed
    turn = len(PREFIX)
    for action in third:
        candidate.step(action)
        turn += 1
        if candidate.levels_completed > base_level or candidate.terminal():
            break
    print("CURRENT_THIRD_PLACED", compact(candidate, turn), flush=True)
    while (
        candidate.levels_completed == base_level
        and not candidate.terminal()
    ):
        candidate.step(5)
        turn += 1
    print("CURRENT_THIRD_RESULT", compact(candidate, turn), flush=True)

    for turn in range(len(PREFIX) + 1, 121):
        clone.step(5)
        if turn in (80, 87, 90, 100, 110, 120):
            print("CURRENT_TAIL", turn, compact(clone, turn), flush=True)
            print(*tile_map(clone.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
