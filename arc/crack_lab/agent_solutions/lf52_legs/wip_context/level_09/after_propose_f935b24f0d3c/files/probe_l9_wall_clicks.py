"""Test coordinate interaction on non-piece level-9 components."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components, safe_step


CONTEXT = int(os.environ.get("CONTEXT_ACTIONS", "0"))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def key(node):
    array = np.asarray(node.frame()).copy()
    array[0, :] = 0
    return array.tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    with open("level9_candidate_102.json") as stream:
        candidate = json.load(stream)
    for action in campaign:
        play(env, action)
    for action in candidate[:CONTEXT]:
        play(env, action)
    root = env.clone()
    blobs = connected_components(root.frame(), colors=(1, 5, 10), min_area=1)
    rows = []
    for index, blob in enumerate(blobs):
        top, left, bottom, right = blob.bbox
        row = max(0, min(63, (top + bottom) // 2))
        col = max(0, min(63, (left + right) // 2))
        click = (6, col, row)
        outcomes = []
        for suffix in ((), (click,), (1,), (2,), (3,), (4,), (7,)):
            node = root.clone()
            play(node, click)
            for action in suffix:
                play(node, action)
            outcomes.append((suffix, key(node) != key(root),
                             int(node.levels_completed)))
        if any(item[1] or item[2] > int(root.levels_completed)
               for item in outcomes):
            rows.append({"index": index, "color": blob.color,
                         "area": blob.area, "bbox": blob.bbox,
                         "click": click, "outcomes": outcomes})
    print("WALL_CLICKS", {"context": CONTEXT,
                          "components": len(blobs), "effects": rows},
          flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
