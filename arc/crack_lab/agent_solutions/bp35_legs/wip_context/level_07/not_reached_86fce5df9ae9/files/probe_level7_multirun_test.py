import json
import os
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from probe_level7_no_control import PREFIX, advance, avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


with open("checkpoint.json") as stream:
    CHECKPOINT_PATH = json.load(stream)["final_path"]


def run_candidate(extra):
    observation = {}

    def probe(env):
        for action in CHECKPOINT_PATH:
            env.step(action)
        gain = advance(env, [*PREFIX, *extra])
        observation.update(
            level=env.levels_completed,
            terminal=env.terminal(),
            gain=gain,
            avatar=None if env.terminal() else avatar_cell(env.frame()),
            controls=() if env.terminal() else controls(env.frame()),
        )

    result = A.run_program("bp35", probe)
    return observation, result


started = time.monotonic()
for extra in [(), ((3,),), ((4,),), ((3,), (4,))]:
    obs, result = run_candidate(extra)
    print("CANDIDATE", extra, obs, "RESULT", result[:2], flush=True)
print("SECONDS", round(time.monotonic() - started, 3))
