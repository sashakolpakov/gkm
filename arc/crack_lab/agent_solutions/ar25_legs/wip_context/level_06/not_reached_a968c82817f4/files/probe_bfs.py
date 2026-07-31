import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import bounded_bfs, level_goal


assert not G._workspace_taint_reason(os.getcwd())
spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def probe(env):
    solver.solve(env)
    base = env.levels_completed
    key = lambda e: np.asarray(e.frame())[:63, :63].tobytes()
    path = bounded_bfs(
        env, level_goal(base), actions=(1, 2, 3, 4, 5),
        key_fn=key, max_states=3000, max_depth=35,
    )
    print("BFS", path)


levels, path, err = A.run_program("ar25", probe)
print("END", levels, len(path), err)
