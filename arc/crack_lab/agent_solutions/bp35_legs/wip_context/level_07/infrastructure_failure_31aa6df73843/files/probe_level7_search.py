import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import click_action, gravity_room_search, run_actions


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    opener = [
        (4,), (4,), (4,), click_action(8, 4),
        (6, 3, 3), (4,), (6, 3, 3), (4,),
    ]
    run_actions(env, opener)
    route = gravity_room_search(
        env.clone(), max_states=300, max_depth=80, support_radius=2,
        edge_gravity=True, debug=False,
    )
    verified = env.clone()
    run_actions(verified, route)
    print("OPENER", opener)
    print("SEARCH", len(route), route)
    print("VERIFY", verified.levels_completed, verified.terminal())


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
