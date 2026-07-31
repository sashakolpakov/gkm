"""Group level-6 coordinate-control effects by avatar context."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
CONTROLS = {
    "A": A,
    "B": B,
    "S": S,
    "DU": (6, 50, 34),
    "DD": (6, 50, 40),
    "DL": (6, 46, 36),
    "DR": (6, 54, 36),
}
PHYSICAL_ENTRY = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def avatar(env):
    frame = np.asarray(env.frame())
    for row in range(0, 62, 2):
        for col in range(0, 40, 2):
            if np.all(frame[row:row + 2, col:col + 2] == 14):
                return row, col
    return None


def movement_paths(env):
    frame = np.asarray(env.frame())
    start = avatar(env)
    queue = deque([start])
    paths = {start: []}
    for_position = ((1, -2, 0), (2, 2, 0), (3, 0, -2), (4, 0, 2))
    while queue:
        row, col = queue.popleft()
        for action, dr, dc in for_position:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            support = sum(
                int(value) not in {0, 4, 5, 15}
                for value in block.flat
            )
            if support < 2:
                continue
            position = nr, nc
            if position not in paths:
                paths[position] = paths[(row, col)] + [action]
                queue.append(position)
    return paths


def normalized_left(env):
    frame = np.asarray(env.frame())[:62, :40].copy()
    position = avatar(env)
    if position is not None:
        row, col = position
        frame[row:row + 2, col:col + 2] = 2
    return frame


def ring(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    )


def solid_exits(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(0, 62, 2)
        for col in range(0, 40, 2)
        if np.all(frame[row:row + 2, col:col + 2] == 11)
    )


def effect_signature(before, child):
    after = normalized_left(child)
    changed = np.argwhere(before != after)
    bbox = None
    if len(changed):
        bbox = (
            int(changed[:, 0].min()),
            int(changed[:, 1].min()),
            int(changed[:, 0].max()),
            int(changed[:, 1].max()),
        )
    return (
        int(len(changed)),
        bbox,
        ring(child),
        solid_exits(child),
        child.levels_completed,
    )


def observe(env):
    solve.solve(env)
    apply(env, PHYSICAL_ENTRY)
    base_level = env.levels_completed
    paths = movement_paths(env)
    actual_paths = {}
    for _, path in paths.items():
        walked = env.clone()
        apply(walked, path)
        actual_paths.setdefault(avatar(walked), path)
    print(
        "CONTEXT_ROOT", avatar(env), "positions", len(actual_paths),
        "ring", ring(env), flush=True,
    )
    for name, control in CONTROLS.items():
        groups = {}
        for position, path in actual_paths.items():
            walked = env.clone()
            apply(walked, path)
            before = normalized_left(walked)
            child = walked.clone()
            step(child, control)
            signature = effect_signature(before, child)
            groups.setdefault(signature, []).append(
                (position, avatar(child))
            )
        print("CONTROL", name, "groups", len(groups), flush=True)
        for signature, entries in groups.items():
            count, bbox, after_ring, exits, level = signature
            if count or level > base_level:
                print(
                    "EFFECT", name, "delta", (count, bbox),
                    "ring", after_ring,
                    "exits", exits, "level", level,
                    "positions", entries, flush=True,
                )


arena.run_program("dc22", observe)
