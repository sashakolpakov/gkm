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


def state(env):
    return (
        env.terminal(),
        tuple(
            (color, tuple(_body_center(group) for group in _body_groups(env, color)))
            for color in (7, 14, 13)
        ),
        tuple(
            (blob.color, blob.bbox)
            for blob in _solid_playfield_squares(env, colors=(8, 15))
        ),
    )


def stage(env, small_target, large_target):
    _click(env, 46, 18)
    for color, target in ((15, small_target), (8, large_target)):
        for _ in range(8):
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
    for small_target in targets:
        for large_target in targets:
            if small_target == large_target:
                continue
            trial = env.clone()
            trial.step(6, 56, 33)
            stage(trial, small_target, large_target)
            staged = state(trial)
            for row, col in ((38, 9), (41, 50), (29, 50), (17, 50)):
                _click(trial, row, col)
            print(
                "LAYOUT", small_target, large_target,
                "STAGED", staged, "MERGED", state(trial),
                flush=True,
            )


print("RUN", A.run_program("su15", program)[0])
