"""Enumerate row ordering around a known winning level-2 x arrangement."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import drive_objects
from players import play_level_1


def moves(start, target):
    return ([1] if target < start else [2]) * (abs(target - start) // 4)


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    play_level_1(env)
    arranged = env.clone()
    drive_objects(arranged, (
        ((14, 18), [4] * 3),
        ((34, 26), [4] * 2),
        ((30, 38), [4] * 2),
    ), commit=[])
    rows = tuple(range(16, 49, 4))
    wins = []
    steps = 0
    started = time.monotonic()
    for a_top, b_top, c_top in itertools.product(rows, repeat=3):
        path = (
            [(6, 40, 26)] + moves(24, b_top)
            + [(6, 30, 38)] + moves(36, c_top)
            + [(6, 21, 18)] + moves(16, a_top)
            + [5]
        )
        result = replay(arranged, path)
        steps += len(path)
        if result.levels_completed > env.levels_completed:
            wins.append((a_top, b_top, c_top))
        target = steps / 280.0
        elapsed = time.monotonic() - started
        if target > elapsed:
            time.sleep(target - elapsed)
    print(
        "L2_ROW_WINS", len(wins), "ROWS", wins[:120], "STEPS", steps,
    )


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
