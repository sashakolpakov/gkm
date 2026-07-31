"""Exhaust columns for the exact notch-row splitter topology."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import POINTS


def moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def path_for(c_left, a_left, b_left, d_left):
    return (
        [(6, *POINTS["A"])] + [4] * 3
        + [(6, *POINTS["B"])] + [3] * 10
        + [(6, *POINTS["C"])] + [1] * 6
        + moves(23, c_left, 3, 4)
        + [(6, *POINTS["D"])] + moves(29, d_left, 3, 4) + [1] * 7
        + [(6, 15, 18)] + [2] * 4 + moves(14, b_left, 3, 4)
        + [(6, 39, 19)] + [2] * 4 + moves(38, a_left, 3, 4)
        + [5]
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    base_level = env.levels_completed
    started = time.monotonic()
    tested = 0
    steps = 0
    grid = tuple(range(5, 51, 3))
    for c_left, a_left, d_left, b_left in itertools.product(
        (17, 20, 23, 26, 29, 32), grid, grid, grid,
    ):
        c_right = c_left + 14
        if not (a_left <= c_right and a_left + 2 >= c_left):
            continue
        if not (d_left + 3 <= c_right and d_left + 5 >= c_left):
            continue
        if not (b_left + 2 < d_left and b_left + 2 < a_left):
            continue
        path = path_for(c_left, a_left, b_left, d_left)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "COLUMN_WIN",
                {
                    "A": (29, a_left),
                    "B": (26, b_left),
                    "C": (14, c_left),
                    "D": (23, d_left),
                },
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("COLUMN_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
