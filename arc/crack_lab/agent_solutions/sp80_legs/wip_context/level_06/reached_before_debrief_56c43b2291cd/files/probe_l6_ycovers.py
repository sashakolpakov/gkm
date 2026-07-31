"""Exhaust all gap-free y covers for the central level-6 x chain."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import covers


STARTS = {"A": (17, 29), "B": (14, 44), "C": (32, 23), "D": (44, 29)}
HEIGHTS = {"A": 6, "B": 12, "C": 3, "D": 6}
ROWS = {
    "A": tuple(range(14, 54, 3)),
    "B": tuple(range(14, 48, 3)),
    "C": tuple(range(14, 57, 3)),
    "D": tuple(range(14, 54, 3)),
}


def axis_moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def gap_free(tops):
    intervals = sorted(
        (tops[piece], tops[piece] + HEIGHTS[piece] - 1)
        for piece in "ABCD"
    )
    end = intervals[0][1]
    for start, stop in intervals[1:]:
        if start > end + 1:
            return False
        end = max(end, stop)
    return intervals[0][0] <= 20 and end >= 43


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def path_for(tops, b_left):
    path = [(6, 25, 33)] + axis_moves(32, tops["C"], 1, 2)
    a_y = 17 if not tops["C"] <= 17 <= tops["C"] + 2 else 20
    path += [(6, 30, a_y)] + axis_moves(17, tops["A"], 1, 2)

    d_point = None
    for point in ((33, 45), (30, 48), (33, 48)):
        x, y = point
        if covers("A", tops["A"], 29, x, y):
            continue
        if covers("C", tops["C"], 23, x, y):
            continue
        d_point = point
        break
    if d_point is None:
        return None
    path += [(6, *d_point)] + axis_moves(44, tops["D"], 1, 2)
    path += (
        [(6, 45, 18)]
        + axis_moves(14, tops["B"], 1, 2)
        + axis_moves(44, b_left, 3, 4)
        + [5]
    )
    return path


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    base_level = env.levels_completed
    started = time.monotonic()
    tested = 0
    steps = 0
    cover_count = 0
    for values in itertools.product(*(ROWS[piece] for piece in "ABCD")):
        tops = dict(zip("ABCD", values))
        if not gap_free(tops):
            continue
        cover_count += 1
        for b_left in (29, 32, 35, 38):
            path = path_for(tops, b_left)
            if path is None:
                continue
            result = replay(env, path)
            tested += 1
            steps += len(path)
            if result.levels_completed > base_level:
                print(
                    "YCOVER_WIN", tops, "B_LEFT", b_left,
                    "TESTED", tested, "STEPS", steps, "PATH", path,
                )
                return
            target = steps / 280.0
            elapsed = time.monotonic() - started
            if target > elapsed:
                time.sleep(target - elapsed)
    print(
        "YCOVER_NONE", "COVERS", cover_count,
        "TESTED", tested, "STEPS", steps,
    )


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
