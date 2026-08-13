"""Fast action-cost search over level 7's final carrier board."""

from heapq import heappop, heappush
import json
import os

import gkm_try
import numpy as np

from perception import arr, safe_step
from probe_undo_slide import groups


LEVEL_START = 316
LEVEL_END = 461
START_GROUP = 17
MAX_STATES = int(os.environ.get("MAX_STATES", "20000"))


def raw(node):
    return arr(node.frame())[1:, :]


def frame_key(node):
    return raw(node).tobytes()


def blocks(grid, color, threshold):
    mask = grid == color
    windows = np.lib.stride_tricks.sliding_window_view(mask, (4, 4))
    sums = windows.sum(axis=(-2, -1))
    candidates = [tuple(point) for point in np.argwhere(sums >= threshold)]
    candidates.sort(key=lambda point: (-int(sums[point]), point))
    selected = []
    for point in candidates:
        if all(abs(point[0] - old[0]) >= 4 or abs(point[1] - old[1]) >= 4 for old in selected):
            selected.append(point)
    return frozenset(selected)


def pieces_grid(grid):
    return blocks(grid, 8, 10), blocks(grid, 14, 12)


def apply_all(node, actions):
    for action in actions:
        safe_step(node, action)


def reconstruct(parent, cursor):
    macros = []
    while parent[cursor] is not None:
        cursor, macro = parent[cursor]; macros.append(macro)
    macros.reverse()
    return tuple(action for macro in macros for action in macro)


def probe(env):
    with open("optimized_prefix_l4_l6_candidate.json") as candidate_file:
        full = json.load(candidate_file)["final_path"]
    apply_all(env, full[:LEVEL_START])
    level_groups = list(groups(full[LEVEL_START:LEVEL_END]))
    keys, pair = level_groups[21]
    level_groups[21] = (keys[:6] + keys[7:], pair)
    for keys, pair in level_groups[:START_GROUP]:
        apply_all(env, keys + pair)
    known_path = tuple(
        action for keys, pair in level_groups[START_GROUP:] for action in keys + pair
    )
    base_level = int(env.levels_completed)

    known = {}; trace = env.clone()
    known[frame_key(trace)] = known_path
    for index, action in enumerate(known_path):
        safe_step(trace, action)
        known.setdefault(frame_key(trace), known_path[index + 1:])

    root = env.clone(); root_key = frame_key(root)
    distance = {root_key: 0}; parent = {root_key: None}; nodes = {root_key: root}
    root_pieces = pieces_grid(raw(root))
    print("L7_FAST_ROOT", root_pieces, len(known_path), flush=True)
    serial = 0; queue = [(2 * (len(root_pieces[1]) - 1), 0, serial, root_key)]
    upper = len(known_path); best_path = known_path; popped = 0

    while queue and popped < MAX_STATES:
        _, cost, _, state_key = heappop(queue)
        if cost != distance.get(state_key):
            continue
        node = nodes.pop(state_key); popped += 1
        suffix = known.get(state_key)
        if suffix is not None and cost + len(suffix) < upper:
            upper = cost + len(suffix)
            best_path = reconstruct(parent, state_key) + suffix
            print("L7_FAST_BOUND", popped, len(distance), upper, flush=True)
        grid = raw(node); bridges, pegs = pieces_grid(grid)
        if cost + 2 * max(0, len(pegs) - 1) >= upper:
            continue

        macros = [((action,),) for action in (1, 2, 3, 4)]
        for source in sorted(bridges | pegs):
            for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                destination = source[0] + dr, source[1] + dc
                if not 0 <= destination[0] <= 60 or not 0 <= destination[1] <= 60:
                    continue
                macros.append((((6, source[1] + 1, source[0] + 1),
                                (6, destination[1] + 1, destination[0] + 1)),))
        for wrapped_macro in macros:
            macro = wrapped_macro[0]
            child = node.clone(); apply_all(child, macro)
            child_grid = raw(child); child_key = child_grid.tobytes()
            child_cost = cost + len(macro)
            if child_key == state_key or child_cost >= distance.get(child_key, upper):
                continue
            if len(macro) == 2:
                if np.any(child_grid == 3) or pieces_grid(child_grid) == (bridges, pegs):
                    continue
            distance[child_key] = child_cost; parent[child_key] = state_key, macro
            if int(child.levels_completed) > base_level:
                if child_cost < upper:
                    upper = child_cost; best_path = reconstruct(parent, child_key)
                    print("L7_FAST_GOAL", popped, len(distance), upper, flush=True)
                continue
            nodes[child_key] = child; serial += 1
            child_pegs = pieces_grid(child_grid)[1]
            heappush(queue, (
                child_cost + 2 * max(0, len(child_pegs) - 1),
                child_cost, serial, child_key,
            ))
        if popped % 1000 == 0:
            print("L7_FAST_PROGRESS", popped, len(distance), cost, upper, flush=True)

    replay = env.clone(); apply_all(replay, best_path)
    print(
        "L7_FAST_RESULT", popped, len(distance), len(best_path),
        int(replay.levels_completed), bool(queue), best_path,
    )


gkm_try.A.run_program("lf52", probe)
