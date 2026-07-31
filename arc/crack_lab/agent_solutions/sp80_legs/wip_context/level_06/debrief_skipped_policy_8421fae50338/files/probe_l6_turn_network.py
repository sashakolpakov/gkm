"""Bounded directed turn/carrier topology for level 6."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_local_l6 import path_for, replay, safe_order


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    steps = 0
    tested = 0
    for c_top, turn_top, (a_left, d_left), b_top, b_left in itertools.product(
        (11, 14, 17, 20, 23, 26),
        (29, 32),
        ((38, 17), (35, 20)),
        tuple(range(11, 48, 3)),
        (14, 17, 20),
    ):
        targets = {
            "A": (turn_top, a_left),
            "B": (b_top, b_left),
            "C": (c_top, 23),
            "D": (turn_top, d_left),
        }
        order = safe_order(targets)
        if order is None:
            continue
        path = path_for(targets, order)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > env.levels_completed:
            print(
                "TURN_NETWORK_WIN", targets, "ORDER", order,
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("TURN_NETWORK_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
