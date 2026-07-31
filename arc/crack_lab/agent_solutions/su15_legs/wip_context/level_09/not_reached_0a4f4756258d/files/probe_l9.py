import importlib.util
import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import color_counts, connected_components, frame_delta
from perception import bounded_replay_bfs
from legs import (
    _body_groups, _body_center, _solid_playfield_squares,
    _click, _move_square_one_step,
)


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def summary(env):
    blobs = [
        (b.color, b.bbox, b.area)
        for b in connected_components(env.frame(), min_area=4)
        if b.color != 3
    ]
    return color_counts(env.frame()), blobs


def playfield(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(
            env.frame(), colors=range(6, 16), min_area=1
        )
        if b.bbox[0] >= 10
    ]


def program(env):
    def body_state(node):
        return tuple(
            (color, tuple(_body_center(group) for group in _body_groups(node, color)))
            for color in (7, 14, 13)
        )

    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions, flush=True)
    counts, blobs = summary(env)
    print("START", (counts, blobs), flush=True)
    print("PLAYFIELD", playfield(env), flush=True)
    rings = [
        tuple(map(round, b.centroid))
        for b in connected_components(env.frame(), colors=(9,), min_area=9)
        if b.bbox[0] >= 10
    ]
    for small_target in rings:
        for large_target in rings:
            if small_target == large_target:
                continue
            trial = env.clone()
            _click(trial, 46, 18)
            for color, target in ((15, small_target), (8, large_target)):
                for _ in range(3):
                    candidates = _solid_playfield_squares(
                        trial, colors=(color,)
                    )
                    if not candidates:
                        break
                    square = candidates[0]
                    if max(
                        abs(round(square.centroid[0]) - target[0]),
                        abs(round(square.centroid[1]) - target[1]),
                    ) <= 1:
                        break
                    _move_square_one_step(trial, square, target)
            print(
                "STAGE", small_target, large_target,
                "LEVEL", trial.levels_completed,
                "SOLIDS", [
                    (b.color, b.bbox)
                    for b in _solid_playfield_squares(
                        trial, colors=(8, 12, 15)
                    )
                ],
                "RINGS", [
                    b.bbox for b in connected_components(
                        trial.frame(), colors=(9,), min_area=9
                    ) if b.bbox[0] >= 10
                ],
                "BODIES", body_state(trial),
                flush=True,
            )
    def key(node):
        solids = tuple(
            (b.color, b.bbox)
            for b in _solid_playfield_squares(
                node, colors=(6, 8, 10, 11, 12, 15)
            )
        )
        return body_state(node), solids

    def actions(node):
        proposed = [(6, 32, 32)]
        for group in _body_groups(node, 7):
            proposed.extend((6, col, row) for row, col in group)
        return proposed

    path = bounded_replay_bfs(
        env,
        lambda node, _: bool(_body_groups(node, 14)),
        actions,
        key_fn=key,
        max_states=1200,
        max_depth=5,
    )
    print("FIRST_BODY_MERGE", path, flush=True)
    if path:
        clone = env.clone()
        for action in path:
            clone.step(*action)
        print("MERGED_STATE", body_state(clone), playfield(clone), flush=True)
        def merged_actions(node):
            proposed = [(6, 32, 32)]
            for color in (7, 14):
                for group in _body_groups(node, color):
                    proposed.extend((6, col, row) for row, col in group)
            return proposed

        path2 = bounded_replay_bfs(
            clone,
            lambda node, _: (
                len(_body_groups(node, 14)) >= 2
                or bool(_body_groups(node, 13))
            ),
            merged_actions,
            key_fn=key,
            max_states=3000,
            max_depth=7,
        )
        print("SECOND_BODY_MERGE", path2, flush=True)
        if path2:
            for action in path2:
                clone.step(*action)
            print("SECOND_STATE", body_state(clone), playfield(clone), flush=True)
            cursor = clone.clone()
            path3 = []
            for _ in range(12):
                choices = []
                for action in merged_actions(cursor):
                    child = cursor.clone()
                    child.step(*action)
                    if _body_groups(child, 13):
                        choices = [(-1, action, child)]
                        break
                    centers = [
                        _body_center(group)
                        for group in _body_groups(child, 14)
                    ]
                    if len(centers) != 2:
                        continue
                    distance = max(
                        abs(centers[0][0] - centers[1][0]),
                        abs(centers[0][1] - centers[1][1]),
                    )
                    choices.append((distance, action, child))
                if not choices:
                    path3 = None
                    break
                distance, action, cursor = min(
                    choices, key=lambda item: item[0]
                )
                path3.append(action)
                if distance == -1:
                    break
            if path3 is not None and not _body_groups(cursor, 13):
                path3 = None
            print("THIRD_BODY_MERGE", path3, flush=True)
            if path3:
                for action in path3:
                    print("THIRD_TRACE", body_state(clone), action, flush=True)
                    clone.step(*action)
                print("THIRD_STATE", "LEVEL", clone.levels_completed,
                      body_state(clone), playfield(clone), flush=True)


print("RUN", A.run_program("su15", program)[0:2:1])
