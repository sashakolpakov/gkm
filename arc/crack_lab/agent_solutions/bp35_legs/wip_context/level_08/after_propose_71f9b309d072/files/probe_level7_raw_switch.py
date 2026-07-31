import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
from probe_level7_coordinate_decode import advance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    node = env.clone()
    route = [*decoded_route(), (3,), (3,), (3,), (3,)]
    height = advance(node, route)
    frame = node.frame()
    blobs = connected_components(frame, colors=(8,), min_area=1)
    print(
        "ROOT", len(route), height, avatar_cell(frame), controls(frame),
        [(blob.bbox, blob.area) for blob in blobs], lattice(frame),
    )
    for blob in blobs:
        r0, c0, r1, c1 = blob.bbox
        candidates = {
            (c0, r0),
            (c1, r1),
            ((c0 + c1) // 2, (r0 + r1) // 2),
        }
        for x, y in sorted(candidates):
            child = node.clone()
            child.step(6, x, y)
            print(
                "CLICK", blob.bbox, (x, y), child.levels_completed,
                child.terminal(),
                None if child.terminal() else avatar_cell(child.frame()),
                [] if child.terminal() else controls(child.frame()),
            )
            if child.levels_completed > 6:
                print("RAW_WIN", [*route, (6, x, y)], flush=True)
                return


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
