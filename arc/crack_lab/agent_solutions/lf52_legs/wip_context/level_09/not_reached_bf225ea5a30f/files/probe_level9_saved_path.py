"""Replay the preserved level-9 candidate from the validated entry clone."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import safe_step


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    with open("level9_walled_mixed_solution.json") as stream:
        candidate = json.load(stream)["actions"]
    for action in prefix:
        safe_step(env, action)
    node = env.clone()
    for action in candidate:
        safe_step(node, action)
    print("SAVED_PATH", {"actions": len(candidate),
                         "level": node.levels_completed,
                         "terminal": node.terminal()})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
