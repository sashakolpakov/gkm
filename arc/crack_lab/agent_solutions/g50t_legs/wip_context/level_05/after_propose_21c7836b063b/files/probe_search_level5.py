import importlib.util
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from legs import clone_after, fast_reach


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("local_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def board(env):
    # Persistent observation: header indicators plus gameplay gates.  Exclude
    # only the bottom replay strip, which is action-history noise.
    return np.asarray(env.frame())[:62, :63]


def dense(env):
    header = np.asarray(env.frame())[:7, :13]
    return (int(np.count_nonzero(header == 9)), int(np.count_nonzero(header == 1)))


def search(start, max_states=300, max_macros=16):
    base = int(start.levels_completed)
    q = deque([(start.clone(), [], 0)])
    seen = {board(start).tobytes()}
    expanded = 0
    while q and expanded < max_states:
        node, prefix, depth = q.popleft()
        expanded += 1
        reward_path, reach = fast_reach(node)
        if reward_path is not None:
            print("FOUND", expanded, depth, len(seen), prefix + reward_path)
            return prefix + reward_path
        if depth >= max_macros:
            continue
        parent_board = board(node).copy()
        kept = []
        for pos, path in sorted(reach.items(), key=lambda item: (len(item[1]), item[0])):
            macro = path + [5]
            child = clone_after(node, macro)
            if int(child.levels_completed) > base:
                print("FOUND_MACRO", expanded, depth, prefix + macro)
                return prefix + macro
            child_board = board(child)
            changed = int(np.count_nonzero(parent_board != child_board))
            if changed == 0:
                continue
            key = child_board.tobytes()
            fresh = key not in seen
            kept.append((pos, len(path), changed, fresh))
            if fresh:
                seen.add(key)
                q.append((child, prefix + macro, depth + 1))
        print("EXPAND", expanded, "depth", depth, "dense", dense(node),
              "states", len(seen), "kept", kept)
    print("NO_PLAN", expanded, len(seen))
    return None


def probe(env):
    solver.solve(env)
    if env.levels_completed != 4:
        print("arrival", env.levels_completed)
        return
    plan = search(env)
    if plan:
        for action in plan:
            env.step(action)


levels, path, err = A.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
