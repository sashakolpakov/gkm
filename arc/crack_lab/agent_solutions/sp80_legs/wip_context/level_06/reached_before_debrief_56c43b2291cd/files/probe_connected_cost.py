"""Search low-cost configurations with connected x/y projections."""
import itertools
import random
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_local_l6 import path_for, replay, safe_order
from probe_l6_conditioned import STARTS


SIZES = {"A": (6, 6), "B": (12, 3), "C": (3, 15), "D": (6, 6)}
ROWS = {
    piece: tuple(range(14, 59 - height + 1, 3))
    for piece, (height, _) in SIZES.items()
}
COLS = {
    piece: tuple(range(5, 59 - width + 1, 3))
    for piece, (_, width) in SIZES.items()
}


def connected(intervals):
    ordered = sorted(intervals)
    end = ordered[0][1]
    for start, stop in ordered[1:]:
        if start > end + 1:
            return False
        end = max(end, stop)
    return True


def axis_configs(values, size_index, fixed=(), cover=None):
    out = []
    for positions in itertools.product(*(values[piece] for piece in "ABCD")):
        intervals = [
            (
                position,
                position + SIZES[piece][size_index] - 1,
            )
            for piece, position in zip("ABCD", positions)
        ] + list(fixed)
        if not connected(intervals):
            continue
        if cover is not None:
            lo, hi = cover
            if min(a for a, _ in intervals) > lo:
                continue
            if max(b for _, b in intervals) < hi:
                continue
        cost = sum(
            abs(position - STARTS[piece][size_index]) // 3
            for piece, position in zip("ABCD", positions)
        )
        out.append((cost, positions))
    return out


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    exact = "--shell" in sys.argv
    xs = axis_configs(COLS, 1, fixed=((29, 31),))
    ys = axis_configs(ROWS, 0, cover=(23, 40))
    directed = "--directed" in sys.argv
    turns = "--turns" in sys.argv
    root_a = "--root-a" in sys.argv or "--root-a-tight" in sys.argv
    root_a_tight = "--root-a-tight" in sys.argv
    if root_a:
        xs = [
            item for item in xs
            if item[1][0] == 29 and item[1][1] >= 35
        ]
        ys = [
            item for item in ys
            if item[1][0] == 17
            and item[1][1] in ((20,) if root_a_tight else (20, 23))
        ]
    if directed or turns:
        xs = [
            item for item in xs
            if item[1][2] == 23
            and item[1][3] <= item[1][2] <= item[1][0]
            and (
                turns
                or item[1][1] <= item[1][3]
            )
        ]
        ys = [
            item for item in ys
            if item[1][0] == 32 and item[1][3] in (23, 38)
        ]
    candidates = []
    sample = "--sample" in sys.argv
    if sample:
        rng = random.Random(806812)
        wanted = 3000
        chosen = set()
        while len(candidates) < wanted:
            xi = rng.randrange(len(xs))
            yi = rng.randrange(len(ys))
            if (xi, yi) in chosen:
                continue
            chosen.add((xi, yi))
            x_cost, x_positions = xs[xi]
            y_cost, y_positions = ys[yi]
            candidates.append((
                x_cost + y_cost, x_positions, y_positions,
            ))
    else:
        for x_cost, x_positions in xs:
            for y_cost, y_positions in ys:
                cost = x_cost + y_cost
                if not (directed or turns or root_a) and (
                    cost > cap or (exact and cost != cap)
                ):
                    continue
                candidates.append((cost, x_positions, y_positions))
    candidates.sort()

    started = time.monotonic()
    base_level = env.levels_completed
    tested = 0
    steps = 0
    for cost, x_positions, y_positions in candidates:
        targets = {
            piece: (top, left)
            for piece, top, left in zip(
                "ABCD", y_positions, x_positions,
            )
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
                "CONNECTED_COST_WIN", "COST", cost, targets,
                "ORDER", order, "TESTED", tested, "STEPS", steps,
                "PATH", path,
            )
            return
        target_elapsed = steps / 280.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print(
        "CONNECTED_COST_NONE", "CAP", cap, "EXACT", exact,
        "CANDIDATES", len(candidates), "TESTED", tested, "STEPS", steps,
    )


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
