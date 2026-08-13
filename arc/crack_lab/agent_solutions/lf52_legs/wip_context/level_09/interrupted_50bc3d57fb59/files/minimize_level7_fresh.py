"""Fresh-arena delta debugging of the validated level-7 action segment."""

import json
import math
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena


PREFIX_END = 331
LEVEL_END = 476
MAX_TRIALS = 200


with open("checkpoint.json") as handle:
    checkpoint = json.load(handle)

full_path = tuple(checkpoint["final_path"])
prefix = full_path[:PREFIX_END]
candidate = full_path[PREFIX_END:LEVEL_END]
suffix = full_path[LEVEL_END:]
trials = 0


def reaches_level_8(segment):
    global trials
    if trials >= MAX_TRIALS:
        return False
    trials += 1
    path = prefix + tuple(segment) + suffix

    def replay(env):
        for action in path:
            if env.terminal():
                break
            env.step(action)

    levels, _, error = arena.run_program("lf52", replay)
    return levels >= 8 and error is None


assert reaches_level_8(candidate)
granularity = 2
while len(candidate) >= 2 and trials < MAX_TRIALS:
    chunk = math.ceil(len(candidate) / granularity)
    reduced = False
    for start in range(0, len(candidate), chunk):
        trial = candidate[:start] + candidate[start + chunk:]
        if trial and reaches_level_8(trial):
            candidate = trial
            granularity = max(2, granularity - 1)
            reduced = True
            print("FRESH_REDUCED", len(candidate), "trials", trials, flush=True)
            break
        if trials >= MAX_TRIALS:
            break
    if reduced:
        continue
    if granularity >= len(candidate):
        break
    granularity = min(len(candidate), granularity * 2)

print("FRESH_MINIMIZED_LENGTH", len(candidate), "trials", trials)
print("FRESH_MINIMIZED_ACTIONS", json.dumps(candidate, separators=(",", ":")))
