"""Map selectable pieces and ordinary jumps at the far loaded frontier."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def pieces(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    return holes, bridges, pegs, fixed, carriers


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    if os.environ.get("L9_UNLOAD") == "1":
        safe_step(env, (6, 23, 37))
        safe_step(env, (6, 11, 37))

    for offset in range(15):
        holes, bridges, pegs, fixed, carriers = pieces(env.frame())
        selectable = []
        for kind, source in (
            tuple(("B", cell) for cell in sorted(bridges))
            + tuple(("P", cell) for cell in sorted(pegs))
            + tuple(("F", cell) for cell in sorted(fixed))
        ):
            child = env.clone()
            before = key(child)
            safe_step(child, (6, source[1] + 1, source[0] + 1))
            if key(child) != before:
                selectable.append((kind, source))
        occupied = bridges | pegs | fixed
        cells = holes | bridges | pegs | fixed | carriers
        candidates = []
        for kind, source in selectable:
            if kind not in ("B", "P"):
                continue
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (midpoint in occupied and destination in cells
                        and destination not in occupied):
                    candidates.append((kind, source, destination))
        print("offset", offset,
              "B", tuple(sorted(bridges)), "P", tuple(sorted(pegs)),
              "F", tuple(sorted(fixed)), "C", tuple(sorted(carriers)),
              "selectable", tuple(selectable),
              "ordinary", tuple(candidates), flush=True)
        safe_step(env, 4)


arena.run_program("lf52", probe)
