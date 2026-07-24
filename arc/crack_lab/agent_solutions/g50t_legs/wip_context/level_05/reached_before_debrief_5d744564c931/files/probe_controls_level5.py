import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from legs import clone_after
from perception import frame_delta, connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("local_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def actors(frame):
    return [(b.color, b.bbox, b.area) for b in connected_components(frame, colors=(1, 8, 9), min_area=2)]


def probe(env):
    solver.solve(env)
    if env.levels_completed != 4:
        print("arrival", env.levels_completed)
        return
    base = env.frame()
    print("base", actors(base))
    sequences = [
        [5, a] for a in (1, 2, 3, 4)
    ] + [
        [5, 5, a] for a in (1, 2, 3, 4)
    ] + [
        [2, 5, a] for a in (1, 2, 3, 4)
    ]
    for seq in sequences:
        child = clone_after(env, seq)
        d = frame_delta(base, child.frame())
        print(seq, "delta", (d["count"], d["bbox"]), "level", child.levels_completed,
              "actors", actors(child.frame()))


levels, path, err = A.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
