"""Test physical-world D-pad affordances across all global phase triples."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
DPAD = {
    "U": (6, 50, 34),
    "D": (6, 50, 40),
    "L": (6, 46, 36),
    "R": (6, 54, 36),
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


def paths(env):
    frame = np.asarray(env.frame())
    start = avatar(env)
    queue = deque([start])
    out = {start: []}
    directions = ((1, -2, 0), (2, 2, 0), (3, 0, -2), (4, 0, 2))
    while queue:
        row, col = queue.popleft()
        for action, dr, dc in directions:
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
            if position not in out:
                out[position] = out[(row, col)] + [action]
                queue.append(position)
    return out


def normalized(env):
    frame = np.asarray(env.frame())[:62, :40].copy()
    position = avatar(env)
    if position is not None:
        row, col = position
        frame[row:row + 2, col:col + 2] = 2
    return frame


def observe(env):
    solve.solve(env)
    apply(env, PHYSICAL_ENTRY)
    base_level = env.levels_completed
    tested = 0
    for a_phase in range(6):
        for b_phase in range(2):
            for s_phase in range(4):
                configured = env.clone()
                apply(
                    configured,
                    [A] * a_phase + [B] * b_phase + [S] * s_phase,
                )
                candidates = paths(configured)
                for predicted, path in candidates.items():
                    if not (
                        24 <= predicted[0] <= 42
                        and predicted[1] <= 20
                    ):
                        continue
                    walked = configured.clone()
                    apply(walked, path)
                    position = avatar(walked)
                    if position != predicted:
                        continue
                    before = normalized(walked)
                    for name, control in DPAD.items():
                        child = walked.clone()
                        step(child, control)
                        after = normalized(child)
                        changed = np.argwhere(before != after)
                        tested += 1
                        if len(changed) or child.levels_completed > base_level:
                            bbox = None if not len(changed) else (
                                int(changed[:, 0].min()),
                                int(changed[:, 1].min()),
                                int(changed[:, 0].max()),
                                int(changed[:, 1].max()),
                            )
                            print(
                                "PHASED_DPAD_EFFECT",
                                (a_phase, b_phase, s_phase),
                                position, name, len(changed), bbox,
                                avatar(child), child.levels_completed,
                                flush=True,
                            )
    print("PHASED_DPAD_DONE", tested, flush=True)


arena.run_program("dc22", observe)
