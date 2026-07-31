import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import frame_delta


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

STAGE3 = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11

CONTROLS = {
    "nw8": (49, 46),
    "ne11": (51, 46),
    "sw10": (49, 48),
    "se7": (51, 48),
}


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def controlled_cells(env):
    frame = env.frame()
    out = []
    for r in range(0, 62, 2):
        for c in range(0, 40, 2):
            block = frame[r:r + 2, c:c + 2]
            colors = tuple(sorted(set(int(v) for v in block.ravel())))
            if any(v in (7, 9, 10, 11, 15) for v in colors):
                out.append((r // 2, c // 2, colors))
    return tuple(out)


def run(env):
    solver.solve(env)
    apply(env, STAGE3)
    print("ROOT", controlled_cells(env), "level", env.levels_completed)
    base = env.frame()
    for name, point in CONTROLS.items():
        child = env.clone()
        child.step(6, *point)
        delta = frame_delta(base, child.frame())
        samples = [sample for sample in delta["samples"] if sample[0] < 62]
        print("ONE", name, controlled_cells(child), "level", child.levels_completed,
              "samples", samples)

    def key(node):
        return node.frame()[:62, :40].tobytes()

    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    summaries = {controlled_cells(env)}
    while queue and len(seen) < 120:
        node, path = queue.popleft()
        if node.levels_completed > 5:
            print("CONTROL_WIN", path)
            return
        if len(path) >= 12:
            continue
        for name, point in CONTROLS.items():
            child = node.clone()
            child.step(6, *point)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + [name]
            summary = controlled_cells(child)
            if summary not in summaries:
                summaries.add(summary)
                print("STATE", child_path, summary, "level", child.levels_completed)
            queue.append((child, child_path))
    print("SEARCH_DONE", len(seen), "summaries", len(summaries))


A.run_program("dc22", run)
