import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS
from probe_level7_coordinate_decode import EARLY_SHIFTED, AMBIGUOUS, advance
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


VARIANTS = [
    (False, False, True, False, True),
    (True, False, True, False, False),
    (True, False, True, False, True),
]

WALKS = [
    [],
    [(3,)], [(3,), (3,)], [(3,), (3,), (3,)],
    [(4,)], [(4,), (4,)], [(4,), (4,), (4,)],
]


def build_route(raw_route, late):
    shifted_steps = EARLY_SHIFTED | {
        step
        for step, shifted in zip(AMBIGUOUS, late)
        if shifted
    }
    route = []
    for step, item in enumerate(raw_route, 1):
        action = (item,) if isinstance(item, int) else tuple(item)
        if (
            step in shifted_steps
            and len(action) == 3
            and action[1] != 3
        ):
            action = (action[0], action[1] + 12, action[2])
        route.append(action)
    return route


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    outcomes = {}
    for late in VARIANTS:
        root = env.clone()
        base_route = [
            *build_route(raw_route, late), (3,), (3,), (3,), (3,)
        ]
        base_height = advance(root, base_route)
        print(
            "ROOT", late, base_height, root.levels_completed,
            root.terminal(),
            None if root.terminal() else avatar_cell(root.frame()),
            [] if root.terminal() else controls(root.frame()),
            "" if root.terminal() else lattice(root.frame()),
        )
        if root.terminal():
            continue
        for pre in WALKS:
            staged = root.clone()
            pre_gain = advance(staged, pre)
            if staged.terminal():
                continue
            for y1 in controls(staged.frame()):
                flipped = staged.clone()
                gain1 = pre_gain + advance(flipped, [(6, 3, y1)])
                if flipped.levels_completed > 6:
                    print("WIN_FIRST", late, pre, y1, flush=True)
                    return
                if flipped.terminal():
                    continue
                first_key = (
                    base_height + gain1,
                    avatar_cell(flipped.frame()),
                    tuple(controls(flipped.frame())),
                )
                outcomes.setdefault(
                    first_key, (late, [*pre, (6, 3, y1)])
                )
                for post in WALKS:
                    middle = flipped.clone()
                    gain2 = gain1 + advance(middle, post)
                    if middle.levels_completed > 6:
                        print(
                            "WIN_WALK", late, pre, y1, post, flush=True
                        )
                        return
                    if middle.terminal():
                        continue
                    middle_key = (
                        base_height + gain2,
                        avatar_cell(middle.frame()),
                        tuple(controls(middle.frame())),
                    )
                    path = [*pre, (6, 3, y1), *post]
                    outcomes.setdefault(middle_key, (late, path))
                    for y2 in controls(middle.frame()):
                        child = middle.clone()
                        gain3 = gain2 + advance(child, [(6, 3, y2)])
                        if child.levels_completed > 6:
                            print(
                                "WIN_SECOND", late, path, y2, flush=True
                            )
                            return
                        if child.terminal():
                            continue
                        child_key = (
                            base_height + gain3,
                            avatar_cell(child.frame()),
                            tuple(controls(child.frame())),
                        )
                        outcomes.setdefault(
                            child_key, (late, [*path, (6, 3, y2)])
                        )
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (-item[0][0], -len(item[0][2])),
    )
    print("NO_FRONTIER_WIN", len(ordered))
    for state, source in ordered[:40]:
        print("OPTION", state, source)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
