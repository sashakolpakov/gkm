"""Diversity-preserving beam from the verified turn-44 level-9 state."""

import numpy as np

import gkm_try

from perception import arr
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import TARGET


def features(env):
    grid = arr(env.frame())
    filled = []
    empty = []
    target_helpers = 0
    target_avatar = 0
    for row, col in sorted(TARGET):
        cell = grid[row * 4:row * 4 + 4, col * 4:col * 4 + 4]
        colors = set(int(value) for value in cell.flat)
        if 4 in colors and 9 in colors:
            filled.append((row, col))
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        target_helpers += 12 in colors
        target_avatar += 14 in colors
    filled = tuple(filled)
    empty = tuple(empty)
    held = int(np.count_nonzero(grid == 0)) > 4
    avatar_pixels = np.argwhere(grid == 14)
    avatar = (
        (99, 99)
        if not len(avatar_pixels)
        else (
            int(avatar_pixels[:, 0].mean()) // 4,
            int(avatar_pixels[:, 1].mean()) // 4,
        )
    )
    target_distance = min(
        (abs(avatar[0] - row) + abs(avatar[1] - col)
         for row, col in empty),
        default=0,
    )
    score = (
        len(filled),
        8 - len(empty),
        held,
        target_helpers,
        -target_avatar,
        -target_distance,
    )
    bucket = (filled, empty, held, target_helpers, avatar)
    return score, bucket, {"empty": empty, "filled": filled}


def search(start, max_depth=25, beam_width=700, bucket_width=8,
           max_transitions=60000):
    base_level = start.levels_completed
    frontier = [(start.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        unique = {}
        best = None
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                if child.levels_completed > base_level:
                    print("TAIL_BEAM_WIN", depth, transitions, child_path,
                          flush=True)
                    return child_path
                score, bucket, target = features(child)
                candidate = (
                    score, child, child_path, bucket, target,
                )
                key = arr(child.frame()).tobytes()
                old = unique.get(key)
                if old is None or score > old[0]:
                    unique[key] = candidate
                if best is None or score > best[0]:
                    best = candidate
                if transitions >= max_transitions:
                    print("TAIL_BEAM_LIMIT", transitions, best[0],
                          best[2], best[4], flush=True)
                    return None

        buckets = {}
        for candidate in unique.values():
            buckets.setdefault(candidate[3], []).append(candidate)
        retained = []
        for candidates in buckets.values():
            candidates.sort(key=lambda item: item[0], reverse=True)
            retained.extend(candidates[:bucket_width])
        retained.sort(key=lambda item: item[0], reverse=True)
        retained = retained[:beam_width]
        frontier = [(item[1], item[2]) for item in retained
                    if not item[1].terminal()]
        print(
            "TAIL_BEAM_DEPTH",
            depth,
            len(frontier),
            len(buckets),
            transitions,
            best[0],
            best[2],
            best[4],
            flush=True,
        )
        if not frontier:
            break
    return None


def inspect(env):
    gkm_try.resumed_solve(env)
    state = env.clone()
    for action in TWO_STAGED + DISMISS:
        state.step(action)
    print("TAIL_BEAM_START", features(state), flush=True)
    print("TAIL_BEAM_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
