"""Scan level 9 after the shortest capture leaves the carrier empty."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


EMPTY_OPENING = (
    ((42, 18), (42, 30)),
    ((48, 24), (36, 24)),
    ((42, 24), (30, 24)),
    ((36, 24), (24, 24)),
    ((30, 24), (18, 24)),
    ((18, 24), (18, 36)),
    ((18, 36), (30, 36)),
    ((24, 36), (36, 36)),
    ((30, 36), (42, 36)),
    ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
    ((48, 30), (36, 30)),
)


def play_move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def pieces(frame):
    out = []
    for blob in connected_components(frame, colors=(9, 11, 12, 14, 15)):
        if blob.color in (9, 12, 14) and blob.size in ((2, 2), (4, 4)):
            if blob.color != 9 or blob.area >= 12:
                out.append((blob.color, blob.top_left, blob.bbox, blob.area))
        elif blob.color in (11, 15) and blob.area >= 4:
            out.append((blob.color, blob.top_left, blob.bbox, blob.area))
    return tuple(out)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in EMPTY_OPENING:
        play_move(env, source, destination)

    for offset in range(16):
        print("empty_scan", offset, pieces(env.frame()), flush=True)
        safe_step(env, 4)


arena.run_program("lf52", probe)
