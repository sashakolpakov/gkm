"""Compare rewind states with independently reconstructed route prefixes."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import avatar_column
from perception import arr, frame_delta
from probe_l8_climb4 import ROOT_ROUTE
from probe_l8_overwall import HANDOFFS, OVERWALL
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def run(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base = env.clone()
    run(base, ROOT_ROUTE)
    base_frame = arr(base.frame()).copy()
    for count in range(1, 17):
        node = env.clone()
        run(node, [*OVERWALL, *([(7,)] * count)])
        delta = frame_delta(base_frame, node.frame())
        print(
            "UNDO",
            count,
            node.levels_completed,
            node.terminal(),
            None if node.terminal() else avatar_column(node.frame()),
            None if node.terminal() else target(node.frame()),
            delta["count"],
            delta["bbox"],
            "" if node.terminal() else lattice(node.frame()),
        )

    exact = env.clone()
    run(exact, [*OVERWALL, *([(7,)] * (len(HANDOFFS) + 2))])
    print(
        "EXACT",
        (arr(exact.frame())[:63] == base_frame[:63]).all(),
        frame_delta(base_frame, exact.frame()),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
