"""Reproduce the validated level-6 bridge-to-carrier attachment geometry."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in connected_components(
            frame, colors=(1, 3, 7, 8, 9, 11, 12, 14, 15)
        )
        if blob.color != 1 or blob.size == (4, 4)
    )


def probe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:258]:
        env.step(action)
    before = arr(env.frame()).copy()
    print("attach_before", int(env.levels_completed), compact(before))
    safe_step(env, (6, 23, 43))
    selected = arr(env.frame()).copy()
    print("attach_selected", frame_delta(before, selected), compact(selected))
    safe_step(env, (6, 35, 43))
    after = arr(env.frame()).copy()
    print("attach_after", frame_delta(before, after), compact(after))


arena.run_program("lf52", probe)
