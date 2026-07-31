"""Sweep side-port row bands for the two endpoint-compatible column layouts."""
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


def path_for(c_top, a_top, a_left, b_top, b_left, d_top, d_left):
    return (
        [(6, *POINTS["A"])] + [4] * 3
        + [(6, *POINTS["B"])] + [3] * 10
        + [(6, *POINTS["C"])] + moves(32, c_top, 1, 2)
        + [(6, 39, 19)] + moves(17, a_top, 1, 2)
        + moves(38, a_left, 3, 4)
        + [(6, 15, 18)] + moves(14, b_top, 1, 2)
        + moves(14, b_left, 3, 4)
        + [(6, *POINTS["D"])] + moves(29, d_left, 3, 4)
        + moves(44, d_top, 1, 2)
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
    columns = (
        (38, 14, 17),
        (35, 17, 20),
    )
    for c_top, (a_left, b_left, d_left), a_top, b_top, d_top in (
        itertools.product(
            (14, 17),
            columns,
            (20, 23, 26, 29, 32, 35),
            (20, 23, 26, 29, 32),
            (20, 23, 26, 29, 32, 35),
        )
    ):
        path = path_for(
            c_top, a_top, a_left, b_top, b_left, d_top, d_left,
        )
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "ROW_WIN",
                {
                    "A": (a_top, a_left),
                    "B": (b_top, b_left),
                    "C": (c_top, 23),
                    "D": (d_top, d_left),
                },
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("ROW_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
