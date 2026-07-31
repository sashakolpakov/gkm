"""Compare pixel-only walk masks with reproduced movement-component sizes."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]
TO_REMOTE_PAD = [4, 1, 4, 1, 4, 4, 1, 1, 1]
REMOTE = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)
ENTRY = (
    [A] * 4
    + TO_BRIDGE
    + [A, 1, 3, A]
    + [1] * 6
    + [B]
    + [1] * 13
    + [S] * 3
    + [2] * 19
    + [A, A, 4, A, 2, A]
    + TO_REMOTE_PAD
    + [B]
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def avatar(frame):
    for row in range(0, 62, 2):
        for col in range(0, 40, 2):
            if np.all(frame[row:row + 2, col:col + 2] == 14):
                return row, col
    return None


def static_reach(frame, blockers):
    start = avatar(frame)
    queue = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            if any(int(value) in blockers for value in block.flat):
                continue
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def uniform_blocker_reach(frame):
    start = avatar(frame)
    queue = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            if any(np.all(block == value) for value in (0, 4, 5, 15)):
                continue
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def dynamic_reach(env):
    queue = deque([env.clone()])
    seen = {avatar(np.asarray(env.frame()))}
    while queue:
        node = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            position = avatar(np.asarray(child.frame()))
            if position not in seen:
                seen.add(position)
                queue.append(child)
    return seen


def report(label, env):
    frame = np.asarray(env.frame())
    print(label, "avatar", avatar(frame), flush=True)
    for name, blockers in (
        ("base", {0, 4, 5, 15}),
        ("ring", {0, 4, 5, 8, 15}),
        ("ring12", {0, 4, 5, 8, 12, 15}),
    ):
        positions = static_reach(frame, blockers)
        print(
            "MASK", label, name, len(positions),
            min(positions), max(positions), flush=True,
        )
    positions = uniform_blocker_reach(frame)
    print(
        "MASK", label, "uniform", len(positions),
        min(positions), max(positions), flush=True,
    )
    if label == "REMOTE":
        actual = dynamic_reach(env)
        print(
            "MASK_DIFF", "base_missing", sorted(actual - static_reach(
                frame, {0, 4, 5, 15}
            )),
            "uniform_extra", sorted(positions - actual),
            flush=True,
        )


def observe(env):
    solve.solve(env)
    remote = env.clone()
    apply(remote, REMOTE)
    report("REMOTE", remote)

    root = env.clone()
    apply(root, ENTRY)
    report("RIGHT3", root)
    hub = root.clone()
    step(hub, B)
    report("HUB", hub)
    right0 = hub.clone()
    apply(right0, [S, B])
    report("RIGHT0", right0)
    top = hub.clone()
    apply(top, [S, S, S, B])
    report("TOP", top)


arena.run_program("dc22", observe)
