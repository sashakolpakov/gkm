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


def route_large_to_upper_left(env):
    for dr, dc in ((-8, -8), (-8, -8), (0, -8)):
        large = _solid_playfield_squares(env, colors=(8,))
        if not large:
            return False
        row, col = map(round, large[0].centroid)
        _click(env, row + dr, col + dc)
    return move_color(env, 8, (41, 11))


def merge_and_stage_small(env):
    pieces = _solid_playfield_squares(env, colors=(6,))
    if len(pieces) < 2:
        return False
    _click(env, pieces[0].bbox[0], pieces[0].bbox[1])
    return move_color(env, 15, (55, 11))


def stage_small_then_large(env):
    if not merge_and_stage_small(env):
        return False
    return route_large_to_upper_left(env)


def snapshot(env):
    return (
        tuple(
            (color, tuple(_body_center(group) for group in _body_groups(env, color)))
            for color in (7, 14, 13)
        ),
        tuple(
            (blob.color, blob.bbox)
            for blob in _solid_playfield_squares(env, colors=(6, 8, 15))
        ),
    )


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)

    for northwest_moves in (0, 1, 2):
        base = env.clone()
        for _ in range(northwest_moves):
            large = _solid_playfield_squares(base, colors=(8,))[0]
            row, col = map(round, large.centroid)
            _click(base, row - 8, col - 8)
        large = _solid_playfield_squares(base, colors=(8,))[0]
        row, col = map(round, large.centroid)
        for dr, dc in (
            (-8, -8), (-8, 0), (-8, 8), (0, -8),
            (0, 8), (8, -8), (8, 0), (8, 8),
        ):
            trial = base.clone()
            _click(trial, row + dr, col + dc)
            print(
                "DIR",
                f"NW_{northwest_moves}",
                (dr, dc),
                snapshot(trial),
                flush=True,
            )

    trace = env.clone()
    print("TRACE", 0, snapshot(trace), flush=True)
    trace_index = 0
    for dr, dc in ((-8, -8), (-8, -8), (0, -8)):
        large = _solid_playfield_squares(trace, colors=(8,))
        if not large:
            break
        row, col = map(round, large[0].centroid)
        _click(trace, row + dr, col + dc)
        trace_index += 1
        print("TRACE", trace_index, "large", snapshot(trace), flush=True)
    large = _solid_playfield_squares(trace, colors=(8,))
    if large:
        _move_square_one_step(trace, large[0], (41, 11))
        trace_index += 1
        print("TRACE", trace_index, "large-target", snapshot(trace), flush=True)
    _click(trace, 46, 18)
    trace_index += 1
    print("TRACE", trace_index, "merge", snapshot(trace), flush=True)
    for _ in range(4):
        small = _solid_playfield_squares(trace, colors=(15,))
        if not small:
            break
        row, col = map(round, small[0].centroid)
        if max(abs(row - 55), abs(col - 11)) <= 1:
            break
        _move_square_one_step(trace, small[0], (55, 11))
        trace_index += 1
        print("TRACE", trace_index, "small", snapshot(trace), flush=True)

    actions = tuple(
        ((6, col, row), _body_center(group), (row, col))
        for group in _body_groups(env, 7)
        for row, col in group
    )
    plain = env.clone()
    plain_ok = stage_small_then_large(plain)
    print("SMALL_FIRST", plain_ok, snapshot(plain), flush=True)
    ranked = []
    for group in _body_groups(plain, 7):
        for row, col in group:
            trial = plain.clone()
            action = (6, col, row)
            trial.step(*action)
            groups7 = _body_groups(trial, 7)
            if len(groups7) < 2:
                print("REDUCE7", action, snapshot(trial), flush=True)
            elif len(groups7) == 2:
                first, second = map(_body_center, groups7)
                distance7 = max(
                    abs(first[0] - second[0]), abs(first[1] - second[1])
                )
                ranked.append((distance7, action, snapshot(trial)))
    for item in sorted(ranked, key=lambda value: (value[0], value[1]))[:6]:
        print("NEAR7", item, flush=True)
    merged7 = plain.clone()
    merged7.step(6, 18, 37)
    reseated = merged7.clone()
    large = _solid_playfield_squares(reseated, colors=(8,))
    if large:
        _move_square_one_step(reseated, large[0], (41, 11))
    print("AFTER_RESEAT", snapshot(reseated), flush=True)
    for group in _body_groups(merged7, 14):
        for row, col in group:
            trial = merged7.clone()
            action = (6, col, row)
            trial.step(*action)
            if _body_groups(trial, 13):
                print("REDUCE14", action, snapshot(trial), flush=True)
    for action, center, point in actions:
        trial = env.clone()
        trial.step(*action)
        staged_ok = stage_small_then_large(trial)
        solids = _solid_playfield_squares(trial, colors=(8, 15))
        if staged_ok and {blob.color for blob in solids} == {8, 15}:
            print(
                "SMALL_FIRST_SAFE",
                center,
                (point[0] - center[0], point[1] - center[1]),
                action,
                snapshot(trial),
                flush=True,
            )
    for action, center, point in actions:
        trial = env.clone()
        trial.step(*action)
        large_ok = route_large_to_upper_left(trial)
        small_ok = merge_and_stage_small(trial)
        solids = _solid_playfield_squares(trial, colors=(8, 15))
        if large_ok and small_ok and {blob.color for blob in solids} == {8, 15}:
            print(
                "SAFE",
                center,
                (point[0] - center[0], point[1] - center[1]),
                action,
                snapshot(trial),
                flush=True,
            )

    base = env.clone()
    for dr, dc in ((-8, -8), (-8, -8), (0, -8)):
        large = _solid_playfield_squares(base, colors=(8,))[0]
        row, col = map(round, large.centroid)
        _click(base, row + dr, col + dc)
    critical_actions = tuple(
        ((6, col, row), _body_center(group), (row, col))
        for group in _body_groups(base, 7)
        for row, col in group
    )
    for action, center, point in critical_actions:
        trial = base.clone()
        trial.step(*action)
        large_ok = move_color(trial, 8, (41, 11))
        small_ok = merge_and_stage_small(trial)
        solids = _solid_playfield_squares(trial, colors=(8, 15))
        if large_ok and small_ok and {blob.color for blob in solids} == {8, 15}:
            print(
                "CRITICAL_SAFE",
                center,
                (point[0] - center[0], point[1] - center[1]),
                action,
                snapshot(trial),
                flush=True,
            )


print("RUN", A.run_program("su15", program)[0])
