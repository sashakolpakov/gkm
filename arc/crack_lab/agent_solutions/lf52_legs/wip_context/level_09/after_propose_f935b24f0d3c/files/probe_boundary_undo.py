"""Check undo behavior immediately after reproduced level transitions."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import frame_delta, safe_step


BOUNDARIES = (8, 42, 87, 149, 238, 331, 476, 544)
TARGET = int(os.environ.get("TARGET_LEVEL", "1"))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def probe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    boundary = BOUNDARIES[TARGET - 1]
    for action in path[:boundary]:
        play(env, action)
    root = env.clone()
    before = root.frame()
    play(root, 7)
    after_undo_level = int(root.levels_completed)
    delta = {key: value for key, value in frame_delta(before, root.frame()).items()
             if key != "samples"}
    replay_one = root.clone()
    play(replay_one, path[boundary - 1])
    replay_two = root.clone()
    for action in path[boundary - 2:boundary]:
        play(replay_two, action)
    print("BOUNDARY_UNDO", {
        "target": TARGET, "boundary": boundary,
        "before_level": int(env.levels_completed),
        "after_undo_level": after_undo_level,
        "delta": delta,
        "replay_one_level": int(replay_one.levels_completed),
        "replay_two_level": int(replay_two.levels_completed),
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
