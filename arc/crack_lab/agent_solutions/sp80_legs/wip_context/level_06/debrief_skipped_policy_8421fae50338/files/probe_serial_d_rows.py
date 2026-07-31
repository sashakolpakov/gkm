"""Sweep row bands for the left-turn-first serial staircase."""
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


def overlaps(a_top, a_height, b_top, b_height):
    return a_top <= b_top + b_height - 1 and b_top <= a_top + a_height - 1


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def path_for(a_top, b_top, c_top, d_top):
    return (
        [(6, *POINTS["B"])] + [3] * 10
        + [(6, *POINTS["A"])] + [4]
        + [(6, *POINTS["D"])] + [3] + moves(44, d_top, 1, 2)
        + [(6, *POINTS["C"])] + moves(32, c_top, 1, 2)
        + [(6, 33, 19)] + moves(17, a_top, 1, 2) + [3] * 3
        + [(6, 15, 18)] + moves(14, b_top, 1, 2) + [4] * 2
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
    for d_top, a_top, c_top, b_top in itertools.product(
        (17, 20, 23, 26, 29),
        (23, 26, 29, 32, 35),
        (32, 35, 38, 41, 44, 47, 50),
        (14, 17, 20, 23, 26, 29, 32, 35),
    ):
        if d_top + 5 >= a_top or a_top + 5 >= c_top:
            continue
        path = path_for(a_top, b_top, c_top, d_top)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "SERIAL_D_WIN",
                {
                    "A": (a_top, 23),
                    "B": (b_top, 20),
                    "C": (c_top, 23),
                    "D": (d_top, 26),
                },
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("SERIAL_D_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
