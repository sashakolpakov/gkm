import json
import os
import sys
from collections import deque

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


def snapshot(env):
    return (
        tuple(
            (color, tuple(_body_center(group) for group in _body_groups(env, color)))
            for color in (7, 14, 13)
        ),
        tuple(
            (blob.color, blob.bbox)
            for blob in _solid_playfield_squares(env, colors=(8, 15))
        ),
    )


def body_actions(env):
    return tuple(
        dict.fromkeys(
            (6, col, row)
            for color in (7, 14, 13)
            for group in _body_groups(env, color)
            for row, col in group
        )
    )


def clearance(env):
    pieces = _solid_playfield_squares(env, colors=(8, 15))
    centers = [
        _body_center(group)
        for color in (7, 14, 13)
        for group in _body_groups(env, color)
    ]
    return min(
        (
            max(
                abs(body_row - round(piece.centroid[0])),
                abs(body_col - round(piece.centroid[1])),
            )
            for body_row, body_col in centers
            for piece in pieces
        ),
        default=0,
    )


def stage(env, small_target, large_target):
    _click(env, 46, 18)
    for color, target in ((15, small_target), (8, large_target)):
        for _ in range(10):
            squares = _solid_playfield_squares(env, colors=(color,))
            if not squares:
                break
            square = squares[0]
            row, col = map(round, square.centroid)
            if max(abs(row - target[0]), abs(col - target[1])) <= 1:
                break
            _move_square_one_step(env, square, target)


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)
    targets = ((41, 11), (55, 11), (55, 53))
    actions = []
    for group in _body_groups(env, 7):
        center = _body_center(group)
        actions.extend(((6, col, row), center) for row, col in group)
    starts = []
    for small_target in targets:
        for large_target in targets:
            if small_target == large_target:
                continue
            for action, original_center in actions:
                trial = env.clone()
                trial.step(*action)
                stage(trial, small_target, large_target)
                pieces = _solid_playfield_squares(trial, colors=(8, 15))
                if {blob.color for blob in pieces} == {8, 15}:
                    print(
                        "SAFE", small_target, large_target,
                        original_center, action, snapshot(trial), flush=True
                    )
                    starts.append((trial, [action]))
    return

    queue = deque(starts)
    seen = {snapshot(node) for node, _ in starts}
    deepest = (0, None, None)
    expanded = 0
    while queue and expanded < 3000:
        node, path = queue.popleft()
        expanded += 1
        if len(path) > deepest[0]:
            deepest = (len(path), path, snapshot(node))
            print("DEPTH", deepest, flush=True)
        if node.levels_completed > env.levels_completed:
            print("WIN", path, snapshot(node), "EXPANDED", expanded, flush=True)
            return
        if len(path) >= 14:
            continue
        for next_action in body_actions(node):
            child = node.clone()
            child.step(*next_action)
            child_pieces = _solid_playfield_squares(child, colors=(8, 15))
            c8 = [blob for blob in child_pieces if blob.color == 8]
            c15 = [blob for blob in child_pieces if blob.color == 15]
            if len(c8) != 1 or not c15 or sum(blob.area for blob in c15) != 9:
                continue
            child_path = path + [next_action]
            child_key = snapshot(child)
            if len(c15) >= 2:
                print("SPLIT", child_path, child_key, flush=True)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    print("DONE", expanded, len(seen), deepest, flush=True)


print("RUN", A.run_program("su15", program)[0])
