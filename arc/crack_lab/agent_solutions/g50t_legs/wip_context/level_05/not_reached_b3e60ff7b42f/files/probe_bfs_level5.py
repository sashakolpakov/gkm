import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import bounded_bfs, level_goal


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("local_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def observation_key(env):
    return np.asarray(env.frame()).tobytes()


def probe(env):
    solver.solve(env)
    if env.levels_completed != 4:
        print("arrival", env.levels_completed)
        return
    plan = bounded_bfs(env, level_goal(4), key_fn=observation_key,
                       max_states=30000, max_depth=100)
    print("plan", plan)
    if plan:
        for action in plan:
            env.step(action)


levels, path, err = A.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
