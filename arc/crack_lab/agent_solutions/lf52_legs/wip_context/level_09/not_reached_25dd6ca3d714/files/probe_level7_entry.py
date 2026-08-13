"""Compact level-7 entry geometry and one-step key responses."""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, color_counts, connected_components, safe_step


def ascii_map(frame):
    data = arr(frame)
    background = max(color_counts(frame), key=color_counts(frame).get)
    glyphs = " .123456789ABCDEF"
    return "\n".join(
        "".join(" " if int(value) == background else glyphs[int(value)]
                for value in data[row])
        for row in range(1, 64)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    prefix = int(os.environ.get("ENTRY_PREFIX", "331"))
    for action in path[:prefix]:
        env.step(action)
    frame = env.frame()
    grouped = defaultdict(list)
    for blob in connected_components(frame, min_area=4):
        grouped[(blob.color, blob.size, blob.area)].append(blob.top_left)
    print("ENTRY", {"level": env.levels_completed,
                    "actions": tuple(env.actions),
                    "colors": color_counts(frame)})
    for signature, positions in sorted(grouped.items()):
        print("COMP", signature, tuple(positions))
    print("MAP\n" + ascii_map(frame))
    print("KEYS")
    base = arr(frame)
    for action in (1, 2, 3, 4):
        child = env.clone()
        safe_step(child, action)
        after = arr(child.frame())
        ys, xs = (base[1:, :] != after[1:, :]).nonzero()
        transitions = Counter(
            (int(base[y + 1, x]), int(after[y + 1, x]))
            for y, x in zip(ys, xs)
        )
        print(action, {"changed": len(ys),
                       "bbox": None if not len(ys) else
                       (int(ys.min() + 1), int(xs.min()),
                        int(ys.max() + 1), int(xs.max())),
                       "transitions": sorted(transitions.items())})
    print("UNDO_CONTEXTS")
    for sequence in ((1, 7), (2, 7), (3, 7), (4, 7),
                     (1, 3, 7), (1, 3, 7, 7),
                     (3, 7, 4), (3, 7, 4, 7),
                     (3, 7, 4, 7, 3),
                     (3, 3, 7), (3, 3, 7, 7), (3, 3, 7, 4)):
        child = env.clone()
        frames = [arr(child.frame())[1:, :].copy()]
        for action in sequence:
            safe_step(child, action)
            frames.append(arr(child.frame())[1:, :].copy())
        matches = tuple(
            next((index for index, prior in enumerate(frames[:-1])
                  if (frames[-1] == prior).all()), None)
            for _ in (0,)
        )
        print(sequence, {"matches_prior_index": matches[0],
                         "matches_root": bool((frames[-1] == frames[0]).all())})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
