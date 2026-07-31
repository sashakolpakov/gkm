import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)


def key(env):
    return np.asarray(env.frame())[:56, :38].tobytes()


def summary(env):
    a = np.asarray(env.frame())[:56, :38]
    targets = int((a == 15).sum())
    ys, xs = np.where(a == 8)
    bbox = None if not len(ys) else (
        int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return targets, bbox


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    seen_frames = {}
    nodes = 0

    def visit(node, path):
        nonlocal nodes
        nodes += 1
        node_key = key(node)
        if node_key not in seen_frames:
            seen_frames[node_key] = path
            print("FRAME", len(seen_frames), "".join(path), summary(node))
        if int(node.levels_completed) > base_level:
            print("REWARD", "".join(path))
            return True
        if len(path) >= 12:
            return False
        for label, action in (("C", C), ("D", D)):
            child = node.clone()
            child.step(*action)
            if visit(child, path + [label]):
                return True
        return False

    visit(env.clone(), [])
    print("DONE", nodes, "FRAMES", len(seen_frames))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
