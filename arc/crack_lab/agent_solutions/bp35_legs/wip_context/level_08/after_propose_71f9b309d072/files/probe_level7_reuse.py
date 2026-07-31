import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import cross_persistent_support_rooms
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    node = env.clone()
    cross_persistent_support_rooms(node)
    print(
        "REUSE", node.levels_completed, node.terminal(),
        None if node.terminal() else avatar_cell(node.frame()),
        [] if node.terminal() else controls(node.frame()),
        "" if node.terminal() else lattice(node.frame()),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
