"""Bounded dense-progress search over the documented level-8 clone surface."""

import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

LEFT_COLLECTOR = (6, 14, 58)
TOP_COLLECTOR = (6, 37, 58)
ACTIONS = (1, 2, 3, 4, LEFT_COLLECTOR, TOP_COLLECTOR)


def apply(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def snapped_center(frame, color):
    points = np.argwhere(np.asarray(frame)[:53] == color)
    row, col = points.mean(axis=0)
    return (
        4.5 + 6 * round((row - 4.5) / 6),
        7.5 + 6 * round((col - 7.5) / 6),
    )


def head_center(frame, color):
    pixels = np.asarray(frame)
    if color == 6:
        points = np.argwhere((pixels[:53, :11] == color))
    else:
        points = np.argwhere((pixels[:8, :] == color))
    row, col = points.mean(axis=0)
    return (
        4.5 + 6 * round((row - 4.5) / 6),
        7.5 + 6 * round((col - 7.5) / 6),
    )


def dense_progress(frame):
    h6 = head_center(frame, 6)
    h15 = head_center(frame, 15)
    centers = {color: snapped_center(frame, color) for color in (8, 9, 12, 14)}
    edges = (
        (h6, centers[9], (0, 6)),
        (centers[9], centers[14], (0, 6)),
        (h15, centers[8], (6, 0)),
        (centers[8], centers[12], (6, 0)),
    )
    errors = []
    for source, target, wanted in edges:
        actual = (target[0] - source[0], target[1] - source[1])
        errors.append(
            (abs(actual[0] - wanted[0]) + abs(actual[1] - wanted[1])) // 6
        )
    horizontal_exact = sum(error == 0 for error in errors[:2])
    vertical_exact = sum(error == 0 for error in errors[2:])
    horizontal_reachable_side = sum(
        centers[color][0] >= h6[0] for color in (9, 14)
    )
    vertical_reachable_side = sum(
        centers[color][1] >= h15[1] for color in (8, 12)
    )
    return (
        horizontal_reachable_side,
        vertical_exact,
        -sum(errors[2:]),
        horizontal_exact,
        -sum(errors[:2]),
        vertical_reachable_side,
        tuple(-error for error in errors),
    )


def search(env, max_depth=9, width=1000):
    base_level = env.levels_completed
    root = env.clone()
    seed = []

    def reconstruct(path):
        node = root.clone()
        for action in path:
            apply(node, action)
        return node

    seeded = reconstruct(seed)
    frontier = [(seed, dense_progress(seeded.frame()))]
    seen = {np.asarray(seeded.frame()).tobytes()}
    print("SEARCH_ENTRY", base_level, frontier[0][1], seed, flush=True)

    for depth in range(len(seed) + 1, max_depth + 1):
        candidates = []
        for path, _ in frontier:
            for action in ACTIONS:
                child_path = path + [action]
                child = reconstruct(child_path)
                if child.levels_completed > base_level:
                    print("FOUND", len(child_path), child_path, flush=True)
                    return child_path
                key = np.asarray(child.frame()).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((child_path, dense_progress(child.frame())))
        candidates.sort(key=lambda item: item[1], reverse=True)
        frontier = candidates[:width]
        best = frontier[0] if frontier else None
        print(
            "DEPTH",
            depth,
            "new",
            len(candidates),
            "seen",
            len(seen),
            "best",
            None if best is None else (best[1], best[0]),
            flush=True,
        )
        if not frontier:
            break
    return None


def probe(env):
    solver.solve(env)
    prefix = [
        TOP_COLLECTOR,
        2,
        3,
        LEFT_COLLECTOR,
        4,
        2,
        4,
        1,
        1,
        TOP_COLLECTOR,
        1,
        LEFT_COLLECTOR,
        2,
        4,
        4,
        4,
        4,
        3,
        3,
        TOP_COLLECTOR,
        4,
        4,
        LEFT_COLLECTOR,
        1,
        1,
        TOP_COLLECTOR,
        2,
        2,
        LEFT_COLLECTOR,
        3,
        TOP_COLLECTOR,
        1,
        1,
        LEFT_COLLECTOR,
        3,
        TOP_COLLECTOR,
        3,
        2,
        2,
        1,
        1,
        4,
        LEFT_COLLECTOR,
        2,
        4,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
    ]
    for action in prefix:
        apply(env, action)
    search(env)


levels, path, err = arena.run_program("sk48", probe)
print("SEARCH_RESULT", levels, len(path), err)
