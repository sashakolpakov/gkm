"""Test every valid coordinate on the one-row action/status strip."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import safe_step


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "9"))
BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def physical(node):
    return np.asarray(node.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:BOUNDARIES[TARGET_LEVEL - 1]]:
        play(env, action)
    root = env.clone()
    results = []
    for x in range(64):
        for count in (1, 2):
            node = root.clone()
            for _ in range(count):
                play(node, (6, x, 0))
            if (physical(node) != physical(root)
                    or node.levels_completed > root.levels_completed):
                results.append((x, count, int(node.levels_completed)))
    print("ROW0", {"level": TARGET_LEVEL, "effects": results}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
