import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import (
    ACTION_NAME,
    action_deltas,
    arr,
    bounded_bfs,
    color_counts,
    level_goal,
    object_candidates,
)


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def summarize(env):
    print("level", int(env.levels_completed), "actions", list(env.actions))
    print("colors", color_counts(env.frame()))
    objects = object_candidates(env.frame())
    print("objects", [(o["color"], o["bbox"], o["area"]) for o in objects])
    deltas = action_deltas(env, [a for a in env.actions if a != 6])
    print(
        "deltas",
        {
            ACTION_NAME.get(a, str(a)): (d["count"], d["bbox"], d["samples"][:8])
            for a, d in deltas.items()
        },
    )


def compact_frame(frame):
    rows = []
    for row in arr(frame):
        runs = []
        start = 0
        for col in range(1, 65):
            if col == 64 or row[col] != row[start]:
                runs.append(f"{start}-{col - 1}:{int(row[start])}")
                start = col
        rows.append(",".join(runs))
    groups = []
    start = 0
    for row in range(1, 65):
        if row == 64 or rows[row] != rows[start]:
            groups.append(f"r{start}-{row - 1} {rows[start]}")
            start = row
    return groups


def trajectory(env):
    def mask(frame, color, r0, c0, height, width):
        grid = arr(frame)
        return "/".join(
            "".join(
                "#" if (grid[r:r + 3, c:c + 3] == color).any() else "."
                for c in range(c0, c0 + width * 3, 3)
            )
            for r in range(r0, r0 + height * 3, 3)
        )

    print("target_mask", mask(env.frame(), 5, 36, 42, 5, 5))
    shaped = env.clone()
    for _ in range(4):
        shaped.step(2)
    print("moving_mask", mask(shaped.frame(), 4, 6, 42, 5, 5))
    print("fixed_mask", mask(env.frame(), 11, 15, 12, 9, 9))
    for actions in ([1], [1] * 4, [2], [2] * 4, [5], [5, 1], [5, 2], [5, 5]):
        clone = env.clone()
        for action in actions:
            clone.step(action)
        objects = object_candidates(clone.frame())
        print(
            "path",
            actions,
            "level",
            int(clone.levels_completed),
            "colors",
            color_counts(clone.frame()),
            "objects",
            [(o["color"], o["bbox"], o["area"]) for o in objects if o["color"] != 9],
        )
    for actions in ([5, 3], [5, 4], [2] * 4 + [5, 1], [2] * 4 + [5, 2],
                    [2] * 4 + [5, 3], [2] * 4 + [5, 4]):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        for action in actions:
            clone.step(action)
        print(
            "context",
            actions[-2:],
            "changed",
            int((before != arr(clone.frame())).sum()),
            "moving4",
            [(o["bbox"], o["area"]) for o in object_candidates(clone.frame()) if o["color"] == 4],
        )
    for action in (3, 4):
        for count in (1, 4, 8, 12, 16):
            clone = env.clone()
            clone.step(5)
            for _ in range(count):
                clone.step(action)
            moving = [
                (o["bbox"], o["area"])
                for o in object_candidates(clone.frame())
                if o["color"] == 4
            ]
            print("side", action, count, "level", int(clone.levels_completed), "moving", moving)
    for count in range(7, 15):
        clone = env.clone()
        for _ in range(count):
            clone.step(2)
        moving = [
            (o["bbox"], o["area"])
            for o in object_candidates(clone.frame())
            if o["color"] == 4
        ]
        print("down", count, "level", int(clone.levels_completed), "moving", moving)
    for actions in (
        [2] * 9 + [5],
        [2] * 9 + [1],
        [2] * 9 + [5, 2],
        [2] * 9 + [5, 1],
        [2] * 9 + [6],
    ):
        clone = env.clone()
        try:
            for action in actions:
                if action == 6:
                    clone.step(6, 10, 43)
                else:
                    clone.step(action)
            print("align_then", actions[-2:], "level", int(clone.levels_completed), "colors", color_counts(clone.frame()))
        except Exception as exc:
            print("align_then", actions[-2:], "error", type(exc).__name__)
    staged = env.clone()
    for _ in range(4):
        staged.step(2)
    for action in (1, 2, 3, 4, 5):
        clone = staged.clone()
        before = arr(clone.frame()).copy()
        clone.step(action)
        delta = action_deltas(staged, [action])[action]
        moving = [
            (o["bbox"], o["area"])
            for o in object_candidates(clone.frame())
            if o["color"] == 4
        ]
        print("staged_action", action, "delta", (delta["count"], delta["bbox"]), "moving", moving)
    for x, y in ((49, 13), (49, 43), (25, 28), (10, 16)):
        clone = staged.clone()
        before = arr(clone.frame()).copy()
        try:
            clone.step(6, x, y)
            changed = int((before != arr(clone.frame())).sum())
            print("click", (x, y), "changed", changed, "level", int(clone.levels_completed))
        except Exception as exc:
            print("click", (x, y), "error", type(exc).__name__)


def mirror_grid_search(env):
    ranked = []
    for down in range(0, 16):
        for right in range(0, 16):
            clone = env.clone()
            for _ in range(down):
                clone.step(2)
            clone.step(5)
            for _ in range(right):
                clone.step(4)
            counts = color_counts(clone.frame())
            if clone.levels_completed > 4:
                print("grid_win", down, right)
                return [2] * down + [5] + [4] * right
            frame = arr(clone.frame())
            interior_12 = int((frame[:, :63] == 12).sum())
            ranked.append((counts.get(4, 0) + interior_12, -counts.get(4, 0), down, right))
    print("grid_dense", sorted(ranked)[:12])
    return None


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solver.solve(env)
    summarize(env)
    print("frame", *compact_frame(env.frame()), sep="\n")
    trajectory(env)
    grid_path = mirror_grid_search(env)
    print("grid_path", grid_path)
    path = bounded_bfs(
        env,
        level_goal(4),
        actions=(1, 2, 3, 4, 5),
        max_states=500,
        max_depth=50,
    )
    print("bfs", path)


A.run_program("ar25", probe)
