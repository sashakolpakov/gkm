"""Measure whether undo cycles gain carrier displacement per real action."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import frame_delta, safe_step


CONTEXT = int(os.environ.get("CONTEXT_ACTIONS", "28"))


def play(env, action):
    safe_step(env, tuple(action) if isinstance(action, list) else action)


def key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    with open("level9_candidate_102.json") as stream:
        candidate = json.load(stream)
    for action in prefix:
        play(env, action)
    root = env.clone()
    for action in candidate[:CONTEXT]:
        play(root, action)

    baselines = {}
    normal = root.clone()
    baselines[key(normal)] = 0
    for count in range(1, 16):
        play(normal, 4)
        baselines.setdefault(key(normal), count)

    experiments = {
        "right": (4,),
        "cycle1": (4, 7, 4),
        "cycle2": (4, 7, 4, 7, 4),
        "cycle3": (4, 7, 4, 7, 4, 7, 4),
        "right2": (4, 4),
        "right_undo_left": (4, 7, 3),
        "right_left_right": (4, 3, 4),
    }
    rows = []
    root_frame = root.frame()
    for name, actions in experiments.items():
        node = root.clone()
        for action in actions:
            play(node, action)
        rows.append({
            "name": name,
            "actions": actions,
            "cost": len(actions),
            "normal_right_match": baselines.get(key(node)),
            "delta": {
                item: value
                for item, value in frame_delta(root_frame, node.frame()).items()
                if item != "samples"
            },
        })
    print("UNDO_PHASE", {"context": CONTEXT, "experiments": rows})


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", {
    "levels": levels,
    "moves": len(path),
    "error": str(error),
})
