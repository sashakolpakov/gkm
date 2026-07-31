import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _body_center, _body_groups, _solid_playfield_squares
from perception import connected_components


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


def load_level(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)


def symbolic(env):
    bodies = tuple(
        (color, tuple(_body_center(group) for group in _body_groups(env, color)))
        for color in (7, 14, 13)
    )
    solids = tuple(
        (blob.color, blob.bbox)
        for blob in _solid_playfield_squares(
            env, colors=(6, 8, 10, 11, 12, 15)
        )
    )
    rings = tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    return bodies, solids, rings


def program(env):
    load_level(env)
    print("START", symbolic(env), flush=True)
    actions = [(6, 32, 32)]
    actions.extend((6, col, row) for row, col in ((46, 18), (52, 23), (51, 38)))
    for group in _body_groups(env, 7):
        actions.extend((6, col, row) for row, col in group)
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        trial = env.clone()
        trial.step(*action)
        print(action, symbolic(trial), flush=True)


print("RUN", A.run_program("su15", program)[0])
