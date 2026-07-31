import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift
from perception import arr
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def clicked_shape(frame, action):
    if len(action) != 3 or action[1] < 15:
        return None
    i = (action[2] - 3) // 6
    j = (action[1] - 15) // 6
    if not (0 <= i < 10 and 0 <= j < 8):
        return None
    return (i, j, _cell_shape(frame, i, j))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    origin = 0
    route = [*decoded_route(), (3,), (3,), (3,), (3,)]
    for step, action in enumerate(route, 1):
        before = arr(env.frame()).copy()
        pre = clicked_shape(before, action)
        env.step(*action)
        after = arr(env.frame())
        shift = 0 if env.terminal() else band_shift(before, after)
        origin -= shift
        changed = int((before[:63] != after[:63]).sum())
        post = None if pre is None or env.terminal() else clicked_shape(after, action)
        print(
            "TRACE", step, action, "world", origin,
            "avatar", None if env.terminal() else avatar_cell(after),
            "shift", shift, "delta", changed,
            "shape", pre, "to", post,
            "controls", () if env.terminal() else tuple(controls(after)),
            "target", False if env.terminal() else bool((after == 7).any()),
            flush=True,
        )
        if env.terminal():
            return


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
