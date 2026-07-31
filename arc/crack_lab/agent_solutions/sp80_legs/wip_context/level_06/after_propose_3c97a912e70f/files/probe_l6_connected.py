"""Search safe level-6 states whose x/y projections are both connected."""
import random
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import ORDER, POINTS, STARTS, path_for, replay, safe_targets


SIZES = {"A": (6, 6), "B": (12, 3), "C": (3, 15), "D": (6, 6)}


def connected(intervals):
    ordered = sorted(intervals)
    end = ordered[0][1]
    for start, stop in ordered[1:]:
        if start > end + 1:
            return False
        end = max(end, stop)
    return True


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    rng = random.Random(806606)
    base_level = env.levels_completed
    started = time.monotonic()
    steps = 0
    tested = 0
    generated = 0

    while tested < 500 and generated < 200000:
        generated += 1
        targets = {}
        for piece, (height, width) in SIZES.items():
            tops = tuple(range(14, 59 - height + 1, 3))
            lefts = tuple(range(5, 59 - width + 1, 3))
            targets[piece] = (rng.choice(tops), rng.choice(lefts))
        x_intervals = [
            (left, left + SIZES[piece][1] - 1)
            for piece, (_, left) in targets.items()
        ] + [(29, 31)]
        y_intervals = [
            (top, top + SIZES[piece][0] - 1)
            for piece, (top, _) in targets.items()
        ]
        if not connected(x_intervals) or not connected(y_intervals):
            continue
        y_min = min(start for start, _ in y_intervals)
        y_max = max(stop for _, stop in y_intervals)
        if y_min > 23 or y_max < 40 or not safe_targets(targets):
            continue

        path = path_for(targets)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "CONNECTED_WIN", targets, "TESTED", tested,
                "GENERATED", generated, "STEPS", steps, "PATH", path,
            )
            return
        target_elapsed = steps / 280.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print(
        "CONNECTED_NONE", tested, "GENERATED", generated, "STEPS", steps,
    )


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
