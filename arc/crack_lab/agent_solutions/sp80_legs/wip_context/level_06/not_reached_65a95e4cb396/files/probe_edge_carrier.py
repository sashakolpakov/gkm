"""Test edge-aligned bottom carriers with safe B-before-C replay."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    arranged = env.clone()
    prefix = (
        [(6, 30, 19)] + moves(17, 29, 1, 2)
        + [(6, 33, 45)] + moves(44, 20, 1, 2) + [3]
    )
    for action in prefix:
        arranged.step(*action) if isinstance(action, tuple) else arranged.step(action)

    tested = 0
    steps = 0
    started = time.monotonic()
    for b_left in range(5, 57, 3):
        for c_top in range(14, 51, 3):
            for c_left in (14, 32):
                c_x = 36 if not b_left <= 36 <= b_left + 2 else 24
                path = (
                    [(6, 45, 18)]
                    + moves(14, 26, 1, 2)
                    + moves(44, b_left, 3, 4)
                    + [(6, c_x, 33)]
                    + moves(32, c_top, 1, 2)
                    + moves(23, c_left, 3, 4)
                    + [5]
                )
                node = arranged.clone()
                valid = True
                for action in path:
                    try:
                        node.step(*action) if isinstance(action, tuple) else node.step(action)
                    except IndexError:
                        valid = False
                        break
                steps += len(path)
                tested += 1
                if valid and node.levels_completed > env.levels_completed:
                    print(
                        "EDGE_WIN", (29, 29), (20, 26),
                        (c_top, c_left), (26, b_left),
                        "PATH", prefix + path,
                    )
                    return
                target = steps / 280.0
                delay = target - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    print("EDGE_NONE", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
