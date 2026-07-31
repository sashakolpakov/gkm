import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from probe_level7_coordinate_decode import advance
from probe_level7_greedy2 import PREFIX, MACRO1
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


KEEP = [(6, 3, 15), (3,), (6, 3, 45)]
NEXT = [(6, 3, 57), (4,), (6, 3, 51)]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    height = advance(root, [*PREFIX, *MACRO1, *KEEP])
    ai, aj = avatar_cell(root.frame())
    supports = [
        (i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
        and _cell_shape(root.frame(), i, j)[1] < 21
    ]
    print(
        "ROOT", height, avatar_cell(root.frame()), controls(root.frame()),
        [(cell, _cell_shape(root.frame(), *cell)) for cell in supports],
        lattice(root.frame()),
    )
    for support in [None, *supports]:
        child = root.clone()
        path = [] if support is None else [click_action(*support)]
        gain = advance(child, [*path, *NEXT])
        print(
            "TEST", support, height + gain, child.levels_completed,
            child.terminal(),
            None if child.terminal() else avatar_cell(child.frame()),
            [] if child.terminal() else controls(child.frame()),
            "" if child.terminal() else lattice(child.frame()),
        )
        if child.levels_completed > 6:
            print("WIN", [*PREFIX, *MACRO1, *KEEP, *path, *NEXT])
            return


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
