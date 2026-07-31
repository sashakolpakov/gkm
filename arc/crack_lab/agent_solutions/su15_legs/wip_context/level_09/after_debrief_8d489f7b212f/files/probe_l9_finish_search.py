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


def move_color(env, color, target):
    for _ in range(8):
        squares = _solid_playfield_squares(env, colors=(color,))
        if not squares:
            return False
        square = squares[0]
        row, col = map(round, square.centroid)
        if max(abs(row - target[0]), abs(col - target[1])) <= 1:
            return True
        _move_square_one_step(env, square, target)
    return False


def stage(env):
    pieces = _solid_playfield_squares(env, colors=(6,))
    _click(env, pieces[0].bbox[0], pieces[0].bbox[1])
    move_color(env, 15, (55, 11))
    for dr, dc in ((-8, -8), (-8, -8), (0, -8)):
        large = _solid_playfield_squares(env, colors=(8,))[0]
        row, col = map(round, large.centroid)
        _click(env, row + dr, col + dc)
    move_color(env, 8, (41, 11))


def summary(env):
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


def key(env):
    return (
        tuple(
            (color, tuple(_body_groups(env, color)))
            for color in (7, 14, 13)
        ),
        summary(env)[1],
    )


def rank(env):
    groups7 = _body_groups(env, 7)
    groups14 = _body_groups(env, 14)
    large = _solid_playfield_squares(env, colors=(8,))
    large_distance = (
        max(
            abs(round(large[0].centroid[0]) - 41),
            abs(round(large[0].centroid[1]) - 11),
        )
        if large else 50
    )
    if len(groups7) == 2 and len(groups14) == 1:
        first, second = map(_body_center, groups7)
        body_cost = 40 + max(
            abs(first[0] - second[0]), abs(first[1] - second[1])
        )
    elif not groups7 and len(groups14) == 2:
        first, second = map(_body_center, groups14)
        body_cost = max(
            abs(first[0] - second[0]), abs(first[1] - second[1])
        )
    else:
        return 1000
    return body_cost + large_distance


def actions(env):
    proposed = [
        (6, col, row)
        for color in (7, 14)
        for group in _body_groups(env, color)
        for row, col in group
    ]
    large = _solid_playfield_squares(env, colors=(8,))
    if large:
        row, col = map(round, large[0].centroid)
        proposed.extend(
            (6, col + dc, row + dr)
            for dr, dc in (
                (-8, -8), (-8, 0), (-8, 8), (0, -8),
                (0, 8), (8, -8), (8, 0), (8, 8),
            )
        )
        proposed.append((6, 11, 41))
    return tuple(dict.fromkeys(proposed))


def deliver_final_body(root, base_level):
    target = (55, 53)
    frontier = [(root.clone(), ())]
    seen = {key(root)}
    for depth in range(1, 27):
        candidates = []
        for node, path in frontier:
            groups = _body_groups(node, 13)
            if not groups:
                continue
            proposed = [(6, col, row) for row, col in groups[0]]
            proposed.extend(((6, 32, 32), (6, 53, 55)))
            for action in tuple(dict.fromkeys(proposed)):
                child = node.clone()
                child.step(*action)
                child_path = path + (action,)
                if child.levels_completed > base_level:
                    print("DELIVERED", child_path, flush=True)
                    return child_path
                groups13 = _body_groups(child, 13)
                if not groups13:
                    continue
                if len(_solid_playfield_squares(child, colors=(8,))) != 1:
                    continue
                if len(_solid_playfield_squares(child, colors=(15,))) != 1:
                    continue
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                row, col = _body_center(groups13[0])
                distance = max(
                    abs(row - target[0]), abs(col - target[1])
                )
                candidates.append(
                    (distance, child_key, child, child_path)
                )
        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = [(item[2], item[3]) for item in candidates[:24]]
        print(
            "DELIVERY_DEPTH",
            depth,
            "BEST",
            candidates[0][0] if candidates else None,
            "KEPT",
            len(frontier),
            flush=True,
        )
        if not frontier:
            return None
    return None


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)
    stage(env)
    print("START", summary(env), flush=True)
    trace = env.clone()
    for action in (
        (6, 16, 39),
        (6, 7, 38),
        (6, 20, 36),
        (6, 21, 34),
        (6, 23, 32),
        (6, 25, 30),
    ):
        point = (action[2], action[1])
        control = None
        for color in (7, 14, 13):
            for group in _body_groups(trace, color):
                if point in group:
                    center = _body_center(group)
                    control = (
                        color,
                        center,
                        (point[0] - center[0], point[1] - center[1]),
                    )
        trace.step(*action)
        print("CONTROL", action, control, summary(trace), flush=True)
    frontier = [(env.clone(), ())]
    seen = {key(env)}
    for depth in range(1, 13):
        candidates = []
        for node, path in frontier:
            for action in actions(node):
                child = node.clone()
                child.step(*action)
                child_path = path + (action,)
                large = _solid_playfield_squares(child, colors=(8,))
                if _body_groups(child, 13) and len(large) == 1:
                    print("MERGED", child_path, summary(child), flush=True)
                    reseat_frontier = [(child.clone(), ())]
                    reseat_seen = {key(child)}
                    for _ in range(5):
                        next_frontier = []
                        for reseat_node, reseat_path in reseat_frontier:
                            groups13 = _body_groups(reseat_node, 13)
                            if not groups13:
                                continue
                            for row, col in groups13[0]:
                                moved = reseat_node.clone()
                                action13 = (6, col, row)
                                moved.step(*action13)
                                moved_large = _solid_playfield_squares(
                                    moved, colors=(8,)
                                )
                                if not moved_large:
                                    continue
                                moved_path = reseat_path + (action13,)
                                trial = moved.clone()
                                _move_square_one_step(
                                    trial, moved_large[0], (41, 11)
                                )
                                final_large = _solid_playfield_squares(
                                    trial, colors=(8,)
                                )
                                if final_large and _body_groups(trial, 13):
                                    final_center = tuple(
                                        map(round, final_large[0].centroid)
                                    )
                                    if max(
                                        abs(final_center[0] - 41),
                                        abs(final_center[1] - 11),
                                    ) <= 1:
                                        print(
                                            "SAFE_RESEAT",
                                            moved_path,
                                            summary(trial),
                                            flush=True,
                                        )
                                        deliver_final_body(
                                            trial, env.levels_completed
                                        )
                                        return
                                moved_key = key(moved)
                                if moved_key in reseat_seen:
                                    continue
                                reseat_seen.add(moved_key)
                                next_frontier.append(
                                    (moved, moved_path)
                                )
                        reseat_frontier = next_frontier[:32]
                    return
                if not large:
                    continue
                groups7 = _body_groups(child, 7)
                groups14 = _body_groups(child, 14)
                if not (
                    (len(groups7) == 2 and len(groups14) == 1)
                    or (not groups7 and len(groups14) == 2)
                ):
                    continue
                if len(_solid_playfield_squares(child, colors=(15,))) != 1:
                    continue
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                candidates.append((rank(child), child_key, child, child_path))
        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = [(item[2], item[3]) for item in candidates[:16]]
        print(
            "DEPTH", depth,
            "BEST", candidates[0][0] if candidates else None,
            "KEPT", len(frontier),
            flush=True,
        )
        if not frontier:
            return


print("RUN", A.run_program("su15", program)[0])
