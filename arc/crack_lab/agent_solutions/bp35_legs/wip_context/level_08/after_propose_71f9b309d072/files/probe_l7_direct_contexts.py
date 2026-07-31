import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
from legs import COL_ANCHORS, ROW_ANCHORS
from probe_level7_reward_recovery import avatar_cell, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


MODES = {
    "base": (),
    "gravity": ((6, 3, 3),),
    "gravity_left": ((6, 3, 3), (3,)),
    "gravity_right": ((6, 3, 3), (4,)),
    "gravity_7": ((6, 3, 3), (7,)),
    "support_gravity": ((6, 27, 15), (6, 3, 3)),
    "four_gravity": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (6, 3, 3),
    ),
    "right_gate": ((4,), (4,), (4,), (4,), (4,), (4,)),
    "cross_gate": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 3, 3), (4,),
    ),
    "cross_rights": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 3, 3), (4,),
        (4,), (4,), (4,), (4,),
    ),
    "cross_lefts": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 3, 3), (4,),
        (3,), (3,), (3,), (3,), (3,), (3,),
    ),
    "cross_right_lefts": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 3, 3), (4,),
        (4,), (4,), (3,), (3,), (3,), (3,), (3,), (3,), (3,),
    ),
    "target_attempt": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (4,), (3,), (3,), (3,), (3,), (3,), (3,),
    ),
    "target_hazard": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 51, 51), (4,), (3,), (3,), (3,), (3,), (3,), (3,),
    ),
    "target_support": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3),
        (4,), (6, 39, 9), (4,), (3,), (3,), (3,), (3,), (3,), (3,),
    ),
    "below_barrier": (
        (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
        (4,), (4,), (4,), (6, 39, 51), (6, 3, 3), (4,),
        (6, 39, 33), (3,), (7,), (4,), (3,),
    ),
}
MODE = os.environ.get("L7_MODE", "base")


def blobs(frame, colour):
    return [
        (blob.bbox, blob.area, tuple(round(v, 1) for v in blob.centroid))
        for blob in connected_components(frame, colors=(colour,), min_area=2)
        if blob.bbox[0] < 63
    ]


def raw_grid(frame):
    symbols = {3: "#", 5: "D", 7: "T", 8: "g", 9: "A", 10: ".",
               11: "a", 12: "s", 14: "Y", 15: "h", 0: "v"}
    return "/".join(
        "".join(symbols.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print(
        "CONTEXT", MODE, 0, "avatar", avatar_cell(env.frame()),
        "agents", blobs(env.frame(), 0), "target", blobs(env.frame(), 7),
        "lattice", raw_grid(env.frame()), flush=True,
    )
    for step, action in enumerate(MODES[MODE], 1):
        env.step(*action)
        print(
            "CONTEXT", MODE, step, action,
            "level", env.levels_completed,
            "terminal", env.terminal(),
            "avatar", None if env.terminal() else avatar_cell(env.frame()),
            "agents", [] if env.terminal() else blobs(env.frame(), 0),
            "target", [] if env.terminal() else blobs(env.frame(), 7),
            "lattice", None if env.terminal() else raw_grid(env.frame()),
            flush=True,
        )
        if env.terminal() or env.levels_completed > 6:
            return


levels, path, error = A.run_program("bp35", probe)
print("RESULT", MODE, levels, len(path), error)
