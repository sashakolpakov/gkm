"""Find exact repeated public states within each reproduced level path."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        path = tuple(normalize(action) for action in json.load(stream)["final_path"])
    level = 1
    seen = {key(env): 0}
    loops = []
    level_start = 0
    for index, action in enumerate(path):
        old_completed = int(env.levels_completed)
        safe_step(env, action)
        state = key(env)
        position = index + 1 - level_start
        if state in seen:
            loops.append((level, seen[state], position, position - seen[state]))
        else:
            seen[state] = position
        if int(env.levels_completed) > old_completed:
            level += 1
            level_start = index + 1
            seen = {key(env): 0}
    print("cycles", tuple(loops), flush=True)


arena.run_program("lf52", probe)
