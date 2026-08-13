"""Find repeated public puzzle frames along each validated level segment."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def profile(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    level = int(env.levels_completed) + 1
    level_start = 0
    seen = {arr(env.frame())[1:, :].tobytes(): 0}
    loops = []
    summaries = []
    for absolute_index, action in enumerate(checkpoint["final_path"], 1):
        safe_step(env, action)
        state = arr(env.frame())[1:, :].tobytes()
        local_index = absolute_index - level_start
        if env.levels_completed >= level:
            summaries.append((level, local_index, tuple(loops)))
            level = int(env.levels_completed) + 1
            level_start = absolute_index
            seen = {state: 0}
            loops = []
            continue
        if state in seen:
            loops.append((seen[state], local_index, local_index - seen[state]))
        else:
            seen[state] = local_index
    print("STATE_LOOPS", tuple(summaries))


levels, path, error = arena.run_program("lf52", profile)
print("PROBE_RESULT", levels, len(path), error)
