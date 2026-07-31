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


LATE = (True, False, True, False, True)
WALKS = [
    [],
    [(3,)], [(3,), (3,)], [(3,), (3,), (3,)],
    [(4,)], [(4,), (4,)], [(4,), (4,), (4,)],
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    route = [
        *build_route(raw_route, LATE),
        (3,), (3,), (3,), (3,),
        (6, 27, 27),
        (3,),
        (6, 3, 27),
    ]
    root = env.clone()
    height = advance(root, route)
    print(
        "ROOT", len(route), height, root.levels_completed, root.terminal(),
        avatar_cell(root.frame()), controls(root.frame()), lattice(root.frame()),
    )
    outcomes = {}
    for pre in WALKS:
        staged = root.clone()
        pre_gain = advance(staged, pre)
        if staged.terminal():
            continue
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            gain1 = pre_gain + advance(flipped, [(6, 3, y1)])
            if flipped.levels_completed > 6:
                print("WIN_FIRST", pre, y1, flush=True)
                return
            if flipped.terminal():
                continue
            path1 = [*pre, (6, 3, y1)]
            outcomes.setdefault(
                (
                    height + gain1,
                    avatar_cell(flipped.frame()),
                    tuple(controls(flipped.frame())),
                    lattice(flipped.frame()),
                ),
                path1,
            )
            for post in WALKS:
                middle = flipped.clone()
                gain2 = gain1 + advance(middle, post)
                if middle.levels_completed > 6:
                    print("WIN_WALK", path1, post, flush=True)
                    return
                if middle.terminal():
                    continue
                path2 = [*path1, *post]
                outcomes.setdefault(
                    (
                        height + gain2,
                        avatar_cell(middle.frame()),
                        tuple(controls(middle.frame())),
                        lattice(middle.frame()),
                    ),
                    path2,
                )
                for y2 in controls(middle.frame()):
                    child = middle.clone()
                    gain3 = gain2 + advance(child, [(6, 3, y2)])
                    if child.levels_completed > 6:
                        print("WIN_SECOND", path2, y2, flush=True)
                        return
                    if child.terminal():
                        continue
                    outcomes.setdefault(
                        (
                            height + gain3,
                            avatar_cell(child.frame()),
                            tuple(controls(child.frame())),
                            lattice(child.frame()),
                        ),
                        [*path2, (6, 3, y2)],
                    )
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (-item[0][0], -len(item[0][2])),
    )
    print("NO_CHAIN_WIN", len(ordered))
    for state, path in ordered:
        print("OPTION", state[:3], path, state[3])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
