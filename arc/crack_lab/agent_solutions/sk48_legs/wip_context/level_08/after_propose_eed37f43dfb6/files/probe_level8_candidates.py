"""Compact candidate-leg checks on a pristine level-8 clone."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import legs

SELECT_LEFT = (6, 14, 58)
SELECT_TOP = (6, 37, 58)


class Recorder:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        self.path = []

    @property
    def levels_completed(self):
        return self.env.levels_completed

    def frame(self):
        return self.env.frame()

    def terminal(self):
        return self.env.terminal()

    def step(self, *action):
        self.path.append(action)
        return self.env.step(*action)


def token_centers(frame):
    pixels = np.asarray(frame)
    result = {}
    for color in (8, 9, 12, 14):
        points = np.argwhere(pixels[:53] == color)
        result[color] = tuple(
            round(float(value), 1) for value in points.mean(axis=0)
        )
    return result


def live_heads(frame):
    pixels = np.asarray(frame)
    left = np.argwhere(pixels[:53, :11] == 6).mean(axis=0)
    top = np.argwhere(pixels[:8, :] == 15).mean(axis=0)
    return {
        6: tuple(round(float(value), 1) for value in left),
        15: tuple(round(float(value), 1) for value in top),
    }


def symbolic(frame):
    pixels = np.asarray(frame)
    rows = []
    for grid_row in range(8):
        chars = []
        for grid_col in range(8):
            row = 2 + 6 * grid_row
            col = 5 + 6 * grid_col
            cell = pixels[row : row + 6, col : col + 6]
            symbol = "."
            for color, candidate in (
                (8, "8"),
                (9, "9"),
                (12, "C"),
                (14, "E"),
                (6, "H"),
                (15, "V"),
            ):
                if np.any(cell == color):
                    symbol = candidate
                    break
            if symbol == "." and any(
                np.any(cell == color) for color in (1, 2, 3)
            ):
                symbol = "+"
            chars.append(symbol)
        rows.append("".join(chars))
    return "/".join(rows)


def apply(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def shortest(root, goal_fn, max_depth=10, max_states=20000):
    actions = (1, 2, 3, 4, SELECT_LEFT, SELECT_TOP)
    queue = deque([(root.clone(), [])])
    seen = {np.asarray(root.frame()).tobytes()}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if goal_fn(node):
            return path, node, len(seen)
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            apply(child, action)
            key = np.asarray(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + [action]))
    return None, None, len(seen)


def probe(env):
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
        for action in checkpoint["final_path"]:
            env.step(action)

    candidate = Recorder(env.clone())
    legs.weave_shared_center_cross(candidate)
    print(
        "CANDIDATE",
        "weave_shared_center_cross",
        len(candidate.path),
        candidate.levels_completed,
        live_heads(candidate.frame()),
        token_centers(candidate.frame()),
        candidate.path,
    )

    seed_path = [
        SELECT_TOP, 2, 3, SELECT_LEFT, 4, 2, 4, 1, 1,
        SELECT_TOP, 1, SELECT_LEFT, 2, 4, 4, 4, 4, 3, 3,
        SELECT_TOP, 4, 4, SELECT_LEFT, 1, 1, SELECT_TOP, 2, 2,
        SELECT_LEFT, 3, SELECT_TOP, 1, 1, SELECT_LEFT, 3,
        SELECT_TOP, 3, 2, 2, 1, 1, 4, SELECT_LEFT, 2, 4, 4,
        4, 4, 3, 3, 3, 3, 3, 3, 2,
    ]
    seeded = env.clone()
    for action in seed_path:
        apply(seeded, action)
    print(
        "SEED55",
        seeded.levels_completed,
        live_heads(seeded.frame()),
        token_centers(seeded.frame()),
        symbolic(seeded.frame()),
    )
    for name, suffix in (
        ("LOWER_AND_EXTEND", [2, 4, 4, 4]),
        ("LOWER_EXTEND_RETRACT", [2, 4, 4, 4, 3, 3, 3]),
        ("RETRACT_SHIFT", [3, 4, 4, 1, 4, 2, 3, 3]),
    ):
        child = seeded.clone()
        trace = []
        for action in suffix:
            apply(child, action)
            trace.append(
                (
                    action,
                    child.levels_completed,
                    token_centers(child.frame()),
                    symbolic(child.frame()),
                )
            )
        print("SUFFIX", name, trace)

    reachable_path, reachable, states = shortest(
        env,
        lambda node: token_centers(node.frame())[14][0] <= 34.5,
        max_depth=10,
        max_states=10000,
    )
    print(
        "REACHABLE14",
        reachable_path,
        states,
        None
        if reachable is None
        else (
            live_heads(reachable.frame()),
            token_centers(reachable.frame()),
            symbolic(reachable.frame()),
        ),
    )


levels, path, err = arena.run_program("sk48", probe)
print("CANDIDATE_RESULT", levels, len(path), err)
