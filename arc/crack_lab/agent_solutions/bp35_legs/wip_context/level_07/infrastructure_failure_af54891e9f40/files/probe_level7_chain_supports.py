import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_decode_frontiers import build_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LATE = (True, False, True, False, True)
HIGH_PREFIXES = [
    [(6, 3, 21)],
    [(4,), (6, 3, 21)],
    [(4,), (6, 3, 21), (4,)],
]
WALKS = [
    [],
    [(3,)], [(3,), (3,)],
    [(4,)], [(4,), (4,)],
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    base_route = [
        *build_route(raw_route, LATE),
        (3,), (3,), (3,), (3,),
        click_action(4, 2),
        (3,),
        (6, 3, 27),
    ]
    base = env.clone()
    base_height = advance(base, base_route)
    outcomes = []
    for high_prefix in HIGH_PREFIXES:
        high = base.clone()
        high_gain = advance(high, high_prefix)
        ai, aj = avatar_cell(high.frame())
        support_cells = [
            (i, j)
            for i in range(max(0, ai - 2), min(10, ai + 3))
            for j in range(max(0, aj - 2), min(8, aj + 3))
            if _cell_shape(high.frame(), i, j)[0] in (12, 14)
        ]
        print(
            "HIGH", high_prefix, base_height + high_gain,
            avatar_cell(high.frame()), controls(high.frame()),
            [(cell, _cell_shape(high.frame(), *cell))
             for cell in support_cells],
            lattice(high.frame()),
        )
        for support in [None, *support_cells]:
            staged = high.clone()
            support_gain = (
                0 if support is None
                else advance(staged, [click_action(*support)])
            )
            if staged.terminal():
                continue
            for walk in WALKS:
                crossed = staged.clone()
                walk_gain = advance(crossed, walk)
                if crossed.terminal():
                    continue
                for y in controls(crossed.frame()):
                    child = crossed.clone()
                    flip_gain = advance(child, [(6, 3, y)])
                    if child.levels_completed > 6:
                        print(
                            "WIN", high_prefix, support, walk, y, flush=True
                        )
                        return
                    if not child.terminal():
                        outcomes.append(
                            (
                                base_height + high_gain + support_gain
                                + walk_gain + flip_gain,
                                high_prefix, support, walk, y,
                                avatar_cell(child.frame()),
                                tuple(controls(child.frame())),
                                lattice(child.frame()),
                            )
                        )
    outcomes.sort(
        key=lambda item: (-len(item[6]), -item[0]),
    )
    print("NO_CHAIN_SUPPORT_WIN", len(outcomes))
    for outcome in outcomes[:50]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
