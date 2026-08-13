"""Reveal L9 after cumulative undo/redo frontier re-entry cycles."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def pieces(frame):
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(7, 9, 11, 12, 14, 15)
        )
        if blob.color not in (9, 12) or blob.area >= 12
    ))


def frame_key(frame):
    return arr(frame)[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in OPENING:
        move(env, source, destination)

    baseline = frame_key(env.frame())
    for cycle in range(9):
        reveal = env.clone()
        for _ in range(9):
            safe_step(reveal, 4)
        print("reset_reveal", cycle,
              frame_key(env.frame()) == baseline,
              int(reveal.levels_completed), pieces(reveal.frame()), flush=True)
        if cycle == 8:
            break
        safe_step(env, 7)
        safe_step(env, 7)
        safe_step(env, (6, 31, 37))
        safe_step(env, (6, 43, 37))


if __name__ == "__main__":
    arena.run_program("lf52", probe)
