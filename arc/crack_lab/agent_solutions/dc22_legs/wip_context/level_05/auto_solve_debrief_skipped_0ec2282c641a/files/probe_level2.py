import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import arr, color_counts, frame_delta, object_candidates
from players import play_level_1


def grid(frame):
    f = arr(frame)
    chars = {2: ".", 4: "#", 6: "6", 7: "7", 8: "8",
             9: "9", 11: "G", 13: "X", 14: "A"}
    lines = []
    for r in range(8, 56, 2):
        line = ""
        for c in range(0, 38, 2):
            vals, counts = __import__("numpy").unique(
                f[r:r + 2, c:c + 2], return_counts=True)
            line += chars.get(int(vals[counts.argmax()]), "?")
        lines.append(line)
    return lines


def probe(env):
    play_level_1(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("grid")
    for i, line in enumerate(grid(env.frame())):
        print(f"{i:02}", line)
    print("objects")
    for obj in object_candidates(env.frame(), min_area=4):
        if obj["color"] != 2:
            print(obj)
    print("deltas")
    base = arr(env.frame()).copy()
    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        delta = frame_delta(base, clone.frame())
        samples = [s for s in delta["samples"] if s[0] < 56]
        print(action, "n", delta["count"], "samples", samples)
    print("clicks")
    click_points = (
        (52, 22), (52, 40), (48, 22), (48, 40),
        (22, 12), (22, 25), (15, 25), (27, 25),
        (9, 29), (17, 41), (11, 41), (23, 41),
    )
    for point in click_points:
        clone = env.clone()
        clone.step(6, *point)
        delta = frame_delta(base, clone.frame())
        print(point, "n", delta["count"], "bbox", delta["bbox"])
        if delta["count"] > 1:
            for i, (a, b) in enumerate(zip(grid(base), grid(clone.frame()))):
                if a != b:
                    print(f"  {i:02}", a, "->", b)
    moves = (1, 2, 3, 4, (6, 52, 22), (6, 52, 40))
    q = deque([(env.clone(), [])])
    seen = {arr(env.frame())[:56].tobytes()}
    solution = None
    best = (10**9, None, None)
    bounds = [99, 99, -1, -1]
    while q and len(seen) < 5000:
        node, path = q.popleft()
        if len(path) >= 70:
            continue
        for move in moves:
            child = node.clone()
            if isinstance(move, tuple):
                child.step(*move)
            else:
                child.step(move)
            new_path = path + [move]
            ys, xs = __import__("numpy").where(arr(child.frame())[:56] == 14)
            if len(ys):
                cy, cx = float(ys.mean()), float(xs.mean())
                distance = abs(cy - 12.5) + abs(cx - 22.5)
                if distance < best[0]:
                    best = (distance, (cy, cx), new_path)
                bounds[0] = min(bounds[0], int(ys.min()))
                bounds[1] = min(bounds[1], int(xs.min()))
                bounds[2] = max(bounds[2], int(ys.max()))
                bounds[3] = max(bounds[3], int(xs.max()))
            if child.levels_completed > 1:
                solution = new_path
                q.clear()
                break
            key = arr(child.frame())[:56].tobytes()
            if key not in seen:
                seen.add(key)
                q.append((child, new_path))
    print("bfs", "states", len(seen), "path", solution,
          "avatar_bounds", bounds, "closest", best)
    staged = env.clone()
    for move in best[2]:
        staged.step(move)
    staged_base = arr(staged.frame()).copy()
    print("staged")
    for point in ((22, 25), (22, 12), (52, 22), (52, 40)):
        child = staged.clone()
        child.step(6, *point)
        delta = frame_delta(staged_base, child.frame())
        print(point, "n", delta["count"], "bbox", delta["bbox"])
    print("select-agent")
    for point in ((22, 12), (6, 30)):
        selected = env.clone()
        selected.step(6, *point)
        before = arr(selected.frame()).copy()
        print("point", point)
        for action in (1, 2, 3, 4):
            child = selected.clone()
            child.step(action)
            delta = frame_delta(before, child.frame())
            samples = [s for s in delta["samples"] if s[0] < 56]
            print(" ", action, samples)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted")
A.run_program("dc22", probe)
