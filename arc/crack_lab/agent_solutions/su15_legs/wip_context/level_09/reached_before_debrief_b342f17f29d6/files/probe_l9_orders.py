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
            return
        square = squares[0]
        row, col = map(round, square.centroid)
        if max(abs(row - target[0]), abs(col - target[1])) <= 1:
            return
        _move_square_one_step(env, square, target)


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


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)
    candidates = tuple(
        ((6, col, row), _body_center(group), (row, col))
        for group in _body_groups(env, 7)
        for row, col in group
    )
    for action, center, point in candidates:
        trial = env.clone()
        trial.step(*action)
        move_color(trial, 8, (55, 11))
        _click(trial, 46, 18)
        move_color(trial, 15, (41, 11))
        solids = _solid_playfield_squares(trial, colors=(8, 15))
        if {blob.color for blob in solids} == {8, 15}:
            print(
                "SAFE", center,
                (point[0] - center[0], point[1] - center[1]),
                action, state(trial),
                flush=True,
            )


print("RUN", A.run_program("su15", program)[0])
