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
    merge_moving_bodies_preserving_cutter,
    reseat_square_while_cutting_staged_square,
    route_cutter_and_merged_body_to_corner_rings,
)
from perception import connected_components


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


def stage(env, small_target, large_target):
    _click(env, 46, 18)
    for color, target in ((15, small_target), (8, large_target)):
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
    targets = tuple(
        tuple(map(round, blob.centroid))
        for blob in connected_components(env.frame(), colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    for small in targets:
        for large in targets:
            if small == large:
                continue
            trial = env.clone()
            stage(trial, small, large)
            staged = state(trial)
            merged = merge_moving_bodies_preserving_cutter(trial)
            after_merge = state(trial)
            cut = reseat_square_while_cutting_staged_square(trial, small)
            after_cut = state(trial)
            routed = route_cutter_and_merged_body_to_corner_rings(trial)
            print(
                "CASE", small, large,
                "STAGED", staged,
                "MERGE", merged, after_merge,
                "CUT", cut, after_cut,
                "ROUTE", routed, state(trial),
                flush=True,
            )


print("RUN", A.run_program("su15", program)[0])
