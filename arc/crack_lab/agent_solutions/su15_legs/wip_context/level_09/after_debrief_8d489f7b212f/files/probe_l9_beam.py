import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    _body_center,
    _body_groups,
    _click,
    _move_square_one_step,
    _solid_playfield_squares,
)


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


def safe_prefix(env):
    env.step(6, 56, 33)
    _click(env, 46, 18)
    for color, target in ((15, (55, 11)), (8, (55, 53))):
        for _ in range(3):
            square = _solid_playfield_squares(env, colors=(color,))[0]
            row, col = map(round, square.centroid)
            if max(abs(row - target[0]), abs(col - target[1])) <= 1:
                break
            _move_square_one_step(env, square, target)
    for row, col in ((38, 9), (41, 50), (29, 50), (17, 50)):
        _click(env, row, col)


def summary(env):
    groups14 = _body_groups(env, 14)
    groups13 = _body_groups(env, 13)
    solids = tuple(
        (blob.color, blob.bbox)
        for blob in _solid_playfield_squares(env, colors=(8, 15))
    )
    state = (
        (
            (14, tuple(_body_center(group) for group in groups14)),
            (13, tuple(_body_center(group) for group in groups13)),
        ),
        solids,
    )
    return state, groups14, groups13


def actions(env):
    return tuple(
        (6, col, row)
        for group in _body_groups(env, 14)
        for row, col in group
    )


def rank(groups, solids):
    solid_penalty = 80 * (
        len([blob for blob in solids if blob[0] == 8]) != 1
        or len([blob for blob in solids if blob[0] == 15]) != 1
    )
    if len(groups) != 2:
        return 500 + solid_penalty
    first, second = map(_body_center, groups)
    dr = abs(first[0] - second[0])
    dc = abs(first[1] - second[1])
    return max(dr, dc) + abs(dr - dc) // 3 + solid_penalty


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)
    safe_prefix(env)
    root = env.clone()
    root_state, _, _ = summary(root)
    print("START", root_state, flush=True)
    frontier = [()]
    seen = {root_state}
    for depth in range(1, 33):
        candidates = []
        for path in frontier:
            node = root.clone()
            for prior in path:
                node.step(*prior)
            for action in actions(node):
                child_path = path + (action,)
                child = root.clone()
                for prior in child_path:
                    child.step(*prior)
                child_state, groups14, groups13 = summary(child)
                if child.levels_completed > env.levels_completed:
                    print("WIN", child_path, child_state, flush=True)
                    return
                if groups13:
                    print("MERGE", child_path, child_state, flush=True)
                    return
                if child_state in seen:
                    continue
                seen.add(child_state)
                candidates.append(
                    (
                        rank(groups14, child_state[1]),
                        child_state,
                        child_path,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = [item[2] for item in candidates[:24]]
        print(
            "DEPTH", depth, "BEST",
            candidates[0][0:2] if candidates else None,
            "KEPT", len(frontier), "SEEN", len(seen),
            flush=True,
        )
        if not frontier:
            break


print("RUN", A.run_program("su15", program)[0])
