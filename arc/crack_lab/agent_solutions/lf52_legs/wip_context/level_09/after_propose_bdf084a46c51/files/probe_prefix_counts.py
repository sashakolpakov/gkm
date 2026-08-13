"""Fresh-replay real action counts at each solved level boundary."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    boundary = 0
    coordinate_actions = 0
    for moves, action in enumerate(path, 1):
        if isinstance(action, list):
            coordinate_actions += 1
        before = int(env.levels_completed)
        env.step(action)
        after = int(env.levels_completed)
        if after > before:
            print("BOUNDARY", {"level": after, "total": moves,
                               "level_moves": moves - boundary,
                               "coordinate_actions": coordinate_actions,
                               "key_actions": moves - boundary -
                               coordinate_actions})
            boundary = moves
            coordinate_actions = 0


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
