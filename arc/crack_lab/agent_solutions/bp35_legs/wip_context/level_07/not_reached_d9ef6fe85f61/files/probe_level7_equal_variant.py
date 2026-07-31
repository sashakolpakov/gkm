import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from probe_level7_coordinate_decode import advance
from probe_level7_decode_frontiers import build_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LATE = (False, True, True, True, False)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    route = build_route(raw_route, LATE)
    root = env.clone()
    height = advance(root, route)
    print(
        "ROOT", len(route), height, avatar_cell(root.frame()),
        controls(root.frame()), lattice(root.frame()),
    )
    for rights in (0, 1):
        staged = root.clone()
        advance(staged, [(4,)] * rights)
        for y in controls(staged.frame()):
            landed = staged.clone()
            gain = advance(landed, [(6, 3, y)])
            print(
                "LAND", rights, y, height + gain,
                landed.levels_completed, landed.terminal(),
                None if landed.terminal() else avatar_cell(landed.frame()),
                [] if landed.terminal() else controls(landed.frame()),
                "" if landed.terminal() else lattice(landed.frame()),
            )
            if landed.levels_completed > 6:
                print(
                    "WIN", [*route, *([(4,)] * rights), (6, 3, y)],
                    flush=True,
                )
                return
            if landed.terminal():
                continue
            for direction in ((3,), (4,)):
                walked = landed.clone()
                walked_gain = 0
                for count in range(4):
                    if walked.levels_completed > 6:
                        print(
                            "WIN_WALK", rights, y, direction, count,
                            flush=True,
                        )
                        return
                    if walked.terminal():
                        break
                    print(
                        "WALK", rights, y, direction, count,
                        height + gain + walked_gain,
                        avatar_cell(walked.frame()),
                        controls(walked.frame()),
                    )
                    walked_gain += advance(walked, [direction])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
