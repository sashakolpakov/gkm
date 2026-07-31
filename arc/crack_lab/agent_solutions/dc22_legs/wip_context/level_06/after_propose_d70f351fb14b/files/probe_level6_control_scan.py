import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A = (6, 56, 8)
B = (6, 51, 25)
S = (6, 51, 48)
REMOTE = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)
ROOT_TO_SELECTOR = (
    [2] * 8
    + [4, 4, A, 4, A, 1]
    + [A, 4] * 3
    + [1, 1, 1]
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def candidates(env):
    points = set()
    for blob in connected_components(env.frame(), min_area=1):
        r0, c0, r1, c1 = blob.bbox
        if c1 < 42 or blob.color in (0, 5):
            continue
        points.add(((c0 + c1) // 2, (r0 + r1) // 2))
        for row in range(r0, r1 + 1, 4):
            for col in range(max(c0, 42), c1 + 1, 4):
                points.add((col, row))
    return sorted(points)


def scan(label, root):
    base = np.asarray(root.frame())[:63].copy()
    groups = {}
    for point in candidates(root):
        child = root.clone()
        child.step(6, *point)
        delta = frame_delta(base, np.asarray(child.frame())[:63])
        if not delta["count"]:
            continue
        changed = base != np.asarray(child.frame())[:63]
        old = base[changed]
        new = np.asarray(child.frame())[:63][changed]
        pairs = {}
        for before, after in zip(old, new):
            pair = (int(before), int(after))
            pairs[pair] = pairs.get(pair, 0) + 1
        signature = (
            delta["count"],
            delta["bbox"],
            tuple(sorted(pairs.items())),
            child.levels_completed,
        )
        groups.setdefault(signature, []).append(point)
    print(label, "groups", groups, flush=True)


def run(env):
    solver.solve(env)
    remote = env.clone()
    apply(remote, REMOTE)
    scan("REMOTE", remote)

    selector = remote.clone()
    apply(selector, ROOT_TO_SELECTOR)
    scan("SELECTOR", selector)

    hub = selector.clone()
    apply(hub, [S, S, S, 3, B])
    scan("HUB", hub)

    top = selector.clone()
    apply(top, [S, S, 3, B])
    scan("TOP", top)


arena.run_program("dc22", run)
