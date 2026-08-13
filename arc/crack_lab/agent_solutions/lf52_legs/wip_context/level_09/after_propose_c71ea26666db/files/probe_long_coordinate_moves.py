"""Test whether visible pieces can jump farther than one lattice interval."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


L9_OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def pieces(frame):
    blobs = connected_components(frame, colors=(1, 8, 9, 12, 14))
    empty = {
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 12)
    }
    sources = {
        ("P" if blob.color == 14 else "B", blob.top_left)
        for blob in blobs
        if (blob.color == 14 and blob.size == (4, 4))
        or (blob.color in (8, 9) and blob.size == (4, 4)
            and blob.area >= 12)
    }
    return tuple(sorted(sources)), tuple(sorted(empty))


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "9"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = env.clone() if desired == 1 else None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current
    prefix = os.environ.get("OPT_KEYS", "")
    if os.environ.get("OPT_L9_OPEN") == "1":
        for source, destination in L9_OPENING:
            safe_step(entry, (6, source[1] + 1, source[0] + 1))
            safe_step(entry, (6, destination[1] + 1,
                              destination[0] + 1))
    if prefix:
        for action in tuple(int(value) for value in prefix.split(",")):
            safe_step(entry, action)

    sources, empty = pieces(entry.frame())
    if os.environ.get("OPT_ALL_GRID") == "1":
        empty = tuple((row, col)
                      for row in range(0, 63, 3)
                      for col in range(0, 63, 3))
    found = []
    tested = 0
    for kind, source in sources:
        for destination in empty:
            distance = (abs(source[0] - destination[0])
                        + abs(source[1] - destination[1]))
            if os.environ.get("OPT_ALL_GRID") == "1":
                pass
            elif os.environ.get("OPT_NONSTANDARD") == "1":
                delta_pair = (abs(source[0] - destination[0]),
                              abs(source[1] - destination[1]))
                if delta_pair in ((0, 12), (12, 0)):
                    continue
            elif distance <= 12:
                continue
            node = entry.clone()
            before = node.frame()
            safe_step(node, (6, source[1] + 1, source[0] + 1))
            safe_step(node, (6, destination[1] + 1,
                             destination[0] + 1))
            tested += 1
            change = delta(before, node.frame())
            if change[0] not in (0, 28):
                found.append((kind, source, destination, distance, change,
                              int(node.levels_completed), pieces(node.frame())))
    print("long_moves", desired, "sources", sources, "empty", len(empty),
          "tested", tested, "found", len(found), flush=True)
    for item in found:
        print("long", item, flush=True)


arena.run_program("lf52", probe)
