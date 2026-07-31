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


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


def state(env):
    return (
        env.levels_completed,
        tuple(
            (color, tuple(_body_center(group) for group in _body_groups(env, color)))
            for color in (7, 14, 13)
        ),
        tuple(
            (blob.color, blob.bbox)
            for blob in _solid_playfield_squares(env, colors=(8, 15))
        ),
    )


def stage(env):
    _click(env, 46, 18)
    for color, target in ((15, (55, 53)), (8, (41, 11))):
        for _ in range(3):
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
    env.step(6, 56, 36)
    stage(env)
    env.step(6, 26, 41)
    print("STAGE", state(env), flush=True)
    for group in _body_groups(env, 14):
        for row, col in group:
            trial = env.clone()
            trial.step(6, col, row)
            groups = _body_groups(trial, 14)
            final = _body_groups(trial, 13)
            distance = None
            if len(groups) == 2:
                centers = [_body_center(item) for item in groups]
                distance = max(
                    abs(centers[0][0] - centers[1][0]),
                    abs(centers[0][1] - centers[1][1]),
                )
            print("NEXT", (6, col, row), distance, bool(final), state(trial), flush=True)


print("RUN", A.run_program("su15", program)[0])
