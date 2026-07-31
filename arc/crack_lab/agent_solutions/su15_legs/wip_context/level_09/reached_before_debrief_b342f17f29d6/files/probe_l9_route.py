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
    _control_body,
    _move_square_one_step,
    _solid_playfield_squares,
)
from perception import connected_components


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


def load_level(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)


def state(env):
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


def nearest_center(env, color, point):
    centers = [_body_center(group) for group in _body_groups(env, color)]
    return min(
        centers,
        key=lambda center: max(
            abs(center[0] - point[0]), abs(center[1] - point[1])
        ),
    )


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


def program(env):
    load_level(env)
    frame = env.frame()
    for row in range(35, 61):
        runs = []
        start = 0
        color = int(frame[row][0])
        for col in range(1, 64):
            next_color = int(frame[row][col])
            if next_color != color:
                runs.append((start, col - 1, color))
                start, color = col, next_color
        runs.append((start, 63, color))
        print("ROW", row, tuple(runs), flush=True)
    print(
        "BARRIERS",
        tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                env.frame(), colors=(4, 5), min_area=1
            )
            if blob.bbox[0] >= 10
        ),
        flush=True,
    )
    safe_prefix(env)
    print("START", state(env), flush=True)
    for index, group in enumerate(_body_groups(env, 14)):
        center = _body_center(group)
        for row, col in group:
            trial = env.clone()
            trial.step(6, col, row)
            print(index, (row - center[0], col - center[1]), state(trial), flush=True)
            groups = _body_groups(trial, 14)
            other = max(groups, key=lambda item: abs(_body_center(item)[1] - col))
            other_center = _body_center(other)
            other_top = min(other)
            trial.step(6, other_top[1], other_top[0])
            moved = nearest_center(trial, 14, (center[0] + 4, center[1]))
            print(
                "VELOCITY", index, (row - center[0], col - center[1]),
                center, moved, other_center,
                flush=True,
            )
    trial = env.clone()
    start_level = trial.levels_completed
    controls = (
        ((-2, 0), lambda groups: max(groups, key=_body_center)),
        ((1, -1), lambda groups: min(groups, key=_body_center)),
        ((0, 2), lambda groups: max(groups, key=_body_center)),
        ((1, -1), lambda groups: min(groups, key=_body_center)),
        ((0, 2), lambda groups: max(groups, key=_body_center)),
        ((1, -1), lambda groups: min(groups, key=_body_center)),
        ((0, 2), lambda groups: max(groups, key=_body_center)),
        ((1, -1), lambda groups: min(groups, key=_body_center)),
        ((0, 2), lambda groups: max(groups, key=_body_center)),
    )
    for control in controls:
        ok = _control_body(trial, start_level, 14, *control)
        print("TRACE", ok, state(trial), flush=True)
        if _body_groups(trial, 13):
            break
    trial = env.clone()
    start_level = trial.levels_completed
    for turn in range(16):
        ok = _control_body(
            trial,
            start_level,
            14,
            (-2, 0),
            lambda groups: max(groups, key=_body_center),
        )
        print("HOLD", turn + 1, ok, state(trial), flush=True)
        if not ok or _body_groups(trial, 13):
            break
    trial = env.clone()
    merge_path = tuple(
        (6, col, 44) for col in range(11, 38, 2)
    ) + ((6, 50, 59), (6, 50, 57), (6, 45, 50))
    for turn, action in enumerate(merge_path, 1):
        trial.step(*action)
        print("MERGEPATH", turn, action, state(trial), flush=True)
    trial = env.clone()
    for action in merge_path[:9]:
        trial.step(*action)
    large = _solid_playfield_squares(trial, colors=(8,))[0]
    _move_square_one_step(trial, large, (55, 45))
    print("DODGE", state(trial), flush=True)
    trial = env.clone()
    for action in merge_path[:8]:
        trial.step(*action)
    large = _solid_playfield_squares(trial, colors=(8,))[0]
    _move_square_one_step(trial, large, (55, 45))
    print("EARLY_DODGE", state(trial), flush=True)
    large = _solid_playfield_squares(trial, colors=(8,))[0]
    _move_square_one_step(trial, large, (55, 53))
    print("RESEAT", state(trial), flush=True)
    start_level = trial.levels_completed
    for turn in range(8):
        ok = _control_body(
            trial,
            start_level,
            14,
            (0, 2),
            lambda groups: min(groups, key=lambda group: _body_center(group)[1]),
        )
        print("CONVERGE", turn + 1, ok, state(trial), flush=True)
        if not ok or _body_groups(trial, 13):
            break
    trial = env.clone()
    for action in merge_path:
        trial.step(*action)
    for group in _body_groups(trial, 13):
        center = _body_center(group)
        print(
            "13_SHAPE",
            center,
            tuple((row - center[0], col - center[1]) for row, col in group),
            flush=True,
        )
    start_level = trial.levels_completed
    for offset, count in (((0, -2), 17), ((-2, 0), 5)):
        for _ in range(count):
            if not _control_body(trial, start_level, 13, offset):
                break
    print("DELIVER13", trial.levels_completed, state(trial), flush=True)
    print(
        "DELIVER_BLOBS",
        tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                trial.frame(), colors=range(6, 16), min_area=1
            )
            if blob.bbox[0] >= 10
        ),
        flush=True,
    )
    trial = env.clone()
    for action in merge_path:
        trial.step(*action)
    start_level = trial.levels_completed
    for offset, count in (((-2, 0), 5), ((0, -2), 17)):
        for _ in range(count):
            if not _control_body(trial, start_level, 13, offset):
                break
    print("DELIVER13_UP_FIRST", trial.levels_completed, state(trial), flush=True)
    print(
        "UP_FIRST_BLOBS",
        tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                trial.frame(), colors=range(6, 16), min_area=1
            )
            if blob.bbox[0] >= 10
        ),
        flush=True,
    )
    trial = env.clone()
    for action in merge_path:
        trial.step(*action)
    start_level = trial.levels_completed
    for offset, count in (((-2, 2), 5), ((0, -2), 22)):
        for index in range(count):
            ok = _control_body(trial, start_level, 13, offset)
            print("DETOUR_TRACE", offset, index + 1, ok, state(trial), flush=True)
            if not ok:
                break
    print("DELIVER13_DETOUR", trial.levels_completed, state(trial), flush=True)
    for group in _body_groups(trial, 13):
        center = _body_center(group)
        print(
            "DETOUR_SHAPE",
            tuple((row - center[0], col - center[1]) for row, col in group),
            flush=True,
        )


print("RUN", A.run_program("su15", program)[0])
