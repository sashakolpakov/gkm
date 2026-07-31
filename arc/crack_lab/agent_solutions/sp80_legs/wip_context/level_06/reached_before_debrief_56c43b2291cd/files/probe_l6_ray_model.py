"""Probe compact level-6 layouts derived from directed beam expansion."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena
import perception as P

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_local_l6 import path_for, replay, safe_order
from probe_l6_conditioned import POINTS


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)

    base_level = env.levels_completed
    started = time.monotonic()
    tested = 0
    steps = 0
    detour_path = (
        [(6, *POINTS["A"])] + [2] * 3 + [4]
        + [(6, *POINTS["C"])] + [1] * 4 + [4]
        + [(6, *POINTS["B"])] + [2] * 4 + [3] * 8
        + [(6, *POINTS["D"])] + [3] * 2 + [1] * 6
        + [5]
    )
    before_commit = replay(env, detour_path[:-1])
    before_frame = before_commit.frame()
    before_commit.step(5)
    after_frame = before_commit.frame()
    print("COMMIT_DELTA", P.frame_delta(before_frame, after_frame))
    pixels_before = P.arr(before_frame)
    pixels_after = P.arr(after_frame)
    for name, (top, left, bottom, right) in {
        "UL": (20, 5, 28, 10),
        "R": (29, 53, 37, 58),
        "LL": (35, 5, 43, 10),
        "B": (53, 26, 58, 34),
    }.items():
        print(
            "TARGET_DELTA", name,
            int((pixels_before[top:bottom + 1, left:right + 1]
                 != pixels_after[top:bottom + 1, left:right + 1]).sum()),
        )
    result = replay(env, detour_path)
    tested += 1
    steps += len(detour_path)
    print(
        "DETOUR_OBJECTS",
        [
            (obj["color"], obj["bbox"], obj["area"])
            for obj in P.object_candidates(result.frame(), min_area=8)
            if obj["color"] in (8, 9, 15)
        ],
    )
    if result.levels_completed > base_level:
        print(
            "RAY_DETOUR_WIN",
            {"A": (26, 32), "B": (26, 20), "C": (20, 26), "D": (26, 23)},
            "PATH", detour_path,
        )
        return

    broad = "--broad" in sys.argv
    for a_top, a_left, d_top, d_left, b_left, c_top in itertools.product(
        (26, 29, 32) if broad else (29,),
        (26, 29, 32) if broad else (29,),
        (20, 23, 26) if broad else (23,),
        (23, 26, 29) if broad else (26,),
        range(8, 24, 3),
        (14, 17, 20),
    ):
        targets = {
            "A": (a_top, a_left),
            "B": (26, b_left),
            "C": (c_top, 23),
            "D": (d_top, d_left),
        }
        order = safe_order(targets)
        if order is None:
            continue
        path = path_for(targets, order)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "RAY_WIN", targets, "ORDER", order,
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("RAY_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
