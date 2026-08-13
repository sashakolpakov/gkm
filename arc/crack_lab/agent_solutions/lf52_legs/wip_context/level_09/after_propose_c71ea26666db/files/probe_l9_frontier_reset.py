"""Test whether undo/redo advances a hidden repeated frontier."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 7, 9, 11, 12, 14, 15))
        if blob.color not in (9, 12) or blob.area >= 12
    )


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    baseline = arr(env.frame()).copy()
    for cycle in range(1, 7):
        safe_step(env, 7)
        safe_step(env, 7)
        safe_step(env, (6, 31, 37))
        safe_step(env, (6, 43, 37))
        print("reset_cycle", cycle, delta(baseline, env.frame()),
              int(env.levels_completed), compact(env.frame()))

    one_undo = env.clone()
    safe_step(one_undo, 7)
    safe_step(one_undo, (6, 43, 37))
    print("single_undo_redo", delta(baseline, one_undo.frame()),
          int(one_undo.levels_completed), compact(one_undo.frame()))


arena.run_program("lf52", probe)
