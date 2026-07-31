"""Compact, observation-only probes for the current sp80 frontier."""
import sys
import random
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import numpy as np
import perception as P
from players import (
    play_level_1,
    play_level_2,
    play_level_3,
    play_level_4,
    play_level_5,
)


def summary(env):
    objects = [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(env.frame(), min_area=4)
    ]
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COLORS", P.color_counts(env.frame()))
    print("OBJECTS", objects)
    print("SHAPES", [
        (o["color"], o["bbox"], tile_shape(env.frame(), o["bbox"]))
        for o in P.object_candidates(env.frame(), min_area=4)
        if o["color"] in (8, 9, 11, 15)
    ])
    print("KEY_DELTAS", {
        action: (delta["count"], delta["bbox"])
        for action, delta in P.action_deltas(env).items()
    })


def tile_shape(frame, bbox):
    f = P.arr(frame)
    r0, c0, r1, c1 = bbox
    rows = []
    for r in range(r0, r1 + 1, 3):
        row = []
        for c in range(c0, c1 + 1, 3):
            vals, counts = np.unique(
                f[r:min(r + 3, r1 + 1), c:min(c + 3, c1 + 1)],
                return_counts=True,
            )
            row.append(int(vals[counts.argmax()]))
        rows.append(tuple(row))
    return tuple(rows)


def layout_runs(frame):
    f = P.arr(frame)
    signatures = []
    for r in range(f.shape[0]):
        runs = []
        start = 0
        for c in range(1, f.shape[1] + 1):
            if c == f.shape[1] or f[r, c] != f[r, start]:
                color = int(f[r, start])
                if color != 12:
                    runs.append((color, start, c - 1))
                start = c
        signatures.append(tuple(runs))
    groups = []
    start = 0
    for r in range(1, len(signatures) + 1):
        if r == len(signatures) or signatures[r] != signatures[start]:
            groups.append((start, r - 1, signatures[start]))
            start = r
    return groups


def active_objects(env):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(env.frame(), min_area=4)
        if o["color"] in (4, 6, 8, 9, 11, 15)
    ]


def replay_actions(env, actions):
    clone = env.clone()
    for action in actions:
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
    return clone


def selection_probes(env):
    points = [
        (30, 19),  # selected center piece
        (45, 18),  # vertical color-8 piece
        (25, 33),  # left color-8 bar
        (34, 33),  # right color-8 bar
        (31, 46),  # color-15 piece
        (7, 22),   # color-11 target
    ]
    for point in points:
        selected = replay_actions(env, [(6, *point)])
        print("SELECT", point, "DELTA",
              (P.frame_delta(env.frame(), selected.frame())["count"],
               P.frame_delta(env.frame(), selected.frame())["bbox"]),
              "OBJECTS", active_objects(selected))
        for action in P.ACTIONS:
            moved = replay_actions(selected, [action])
            delta = P.frame_delta(selected.frame(), moved.frame())
            print(" AFTER", action, (delta["count"], delta["bbox"]),
                  "LEVEL", moved.levels_completed,
                  "OBJECTS", active_objects(moved))


def candidate_probes(env):
    candidates = {
        "D_up2": [(6, 31, 46), 1, 1, 5],
        "D_up3": [(6, 31, 46), 1, 1, 1, 5],
        "A_down1": [(6, 30, 19), 2, 5],
        "A_down6": [(6, 30, 19), 2, 2, 2, 2, 2, 2, 5],
        "projection_chain": (
            [(6, 45, 18)] + [3] * 3
            + [(6, 25, 33)] + [1] * 3
            + [(6, 33, 45)] + [1] * 7
            + [5]
        ),
        "spanning_chain_overlap": (
            [(6, 30, 19)] + [2] * 6
            + [(6, 45, 18)] + [2] * 3 + [3] * 3
            + [(6, 33, 45)] + [1] * 3
            + [5]
        ),
        "spanning_chain_stagger": (
            [(6, 30, 19)] + [2] * 6
            + [(6, 45, 18)] + [2] * 3 + [3] * 3
            + [(6, 33, 45)] + [1] * 2
            + [5]
        ),
    }
    for name, path in candidates.items():
        result = replay_actions(env, path)
        print("CANDIDATE", name, "LEVEL", result.levels_completed,
              "OBJECTS", active_objects(result))


def axis_moves(start, target, negative, positive):
    action = negative if target < start else positive
    return [action] * (abs(target - start) // 3)


def projection_score(env):
    f = P.arr(env.frame())
    moving = np.isin(f, (4, 8, 9, 15))
    side_rows = (24, 33, 39)
    bottom_col = 30
    return sum(bool(moving[r, 11:53].any()) for r in side_rows) + bool(
        moving[8:53, bottom_col].any()
    )


def search_rows(env, trials=600):
    rng = random.Random(806)
    starts = (17, 14, 32, 44)
    choices = (
        tuple(range(2, 39, 3)),
        tuple(range(2, 51, 3)),
        tuple(range(2, 42, 3)),
        tuple(range(2, 51, 3)),
    )
    started = time.monotonic()
    steps = 0
    best = (-1, None)
    for trial in range(trials):
        tops = tuple(rng.choice(options) for options in choices)
        path = (
            [(6, 30, 19)] + axis_moves(17, tops[0], 1, 2)
            + [(6, 45, 18)] + axis_moves(14, tops[1], 1, 2)
            + [(6, 25, 33)] + axis_moves(32, tops[2], 1, 2)
            + [(6, 33, 45)] + axis_moves(44, tops[3], 1, 2)
        )
        node = replay_actions(env, path)
        score = projection_score(node)
        if score > best[0]:
            best = (score, tops)
        node.step(5)
        steps += len(path) + 1
        if node.levels_completed > env.levels_completed:
            print("ROW_WIN", trial, tops, path + [5], "STEPS", steps)
            return
        target_elapsed = steps / 300.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print("ROW_NONE", trials, "BEST", best, "STEPS", steps)


def covers(kind, top, left, x, y):
    rr, cc = y - top, x - left
    if kind == "A":
        return (0 <= rr < 3 and 0 <= cc < 3) or (
            3 <= rr < 6 and 0 <= cc < 6
        )
    if kind == "D":
        return (0 <= rr < 3 and 3 <= cc < 6) or (
            3 <= rr < 6 and 0 <= cc < 6
        )
    width, height = (3, 12) if kind == "B" else (15, 3)
    return 0 <= rr < height and 0 <= cc < width


def search_joint(env, trials=1600):
    rng = random.Random(80664)
    specs = (
        ("A", (30, 21), 17, 29, tuple(range(14, 45, 3)), tuple(range(5, 54, 3))),
        ("B", (45, 18), 14, 44, tuple(range(14, 45, 3)), tuple(range(5, 57, 3))),
        ("C", (25, 33), 32, 23, tuple(range(14, 45, 3)), tuple(range(5, 45, 3))),
        ("D", (33, 45), 44, 29, tuple(range(14, 45, 3)), tuple(range(5, 54, 3))),
    )
    started = time.monotonic()
    steps = 0
    best = (-1, None)
    for trial in range(trials):
        destinations = []
        valid = True
        for index, (kind, point, _, _, rows, cols) in enumerate(specs):
            top, left = rng.choice(rows), rng.choice(cols)
            later_points = [s[1] for s in specs[index + 1:]]
            if any(covers(kind, top, left, x, y) for x, y in later_points):
                valid = False
                break
            destinations.append((top, left))
        if not valid:
            continue
        path = []
        for spec, (top, left) in zip(specs, destinations):
            _, (x, y), start_r, start_c, _, _ = spec
            path += [(6, x, y)]
            path += axis_moves(start_r, top, 1, 2)
            path += axis_moves(start_c, left, 3, 4)
        node = replay_actions(env, path)
        score = projection_score(node)
        if score > best[0]:
            best = (score, tuple(destinations))
        node.step(5)
        steps += len(path) + 1
        if node.levels_completed > env.levels_completed:
            print("JOINT_WIN", trial, tuple(destinations), path + [5],
                  "STEPS", steps)
            return
        target_elapsed = steps / 300.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print("JOINT_NONE", trials, "BEST", best, "STEPS", steps)


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3):
        player(env)
    if "--level4" in sys.argv:
        print("LEVEL4_START")
        summary(env)
        path = (
            [(6, 24, 18)] + [4] * 3
            + [(6, 45, 18)] + [3] * 8
            + [(6, 18, 30)] + [4] * 8
            + [(6, 49, 33)] + [3] * 10
            + [(6, 43, 42)] + [3] * 7
        )
        print("LEVEL4_ARRANGED")
        summary(replay_actions(env, path))
        return
    play_level_4(env)
    if "--level5" in sys.argv:
        print("LEVEL5_START")
        summary(env)
        path = (
            [(6, 48, 33)] + [3] * 6
            + [(6, 21, 33)] + [2] * 3
            + [(6, 34, 21)] + [4] * 3 + [2] * 7
            + [(6, 36, 45)] + [3] * 6 + [1] * 2
        )
        print("LEVEL5_ARRANGED")
        summary(replay_actions(env, path))
        return
    play_level_5(env)
    summary(env)
    if "--layout" in sys.argv:
        print("LAYOUT", layout_runs(env.frame()))
    if "--selections" in sys.argv:
        selection_probes(env)
    if "--candidates" in sys.argv:
        candidate_probes(env)
    if "--search-rows" in sys.argv:
        search_rows(env)
    if "--search-joint" in sys.argv:
        search_joint(env)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
