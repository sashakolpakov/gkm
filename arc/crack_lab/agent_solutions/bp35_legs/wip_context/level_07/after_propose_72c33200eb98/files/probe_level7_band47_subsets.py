import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, click_action
from probe_level7_best_trace import ROUTE
from probe_level7_no_control import PREFIX, advance, avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PATTERNS = [
    ((3,),) * 8,
    ((4,),) * 8,
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_gain = advance(env, [*PREFIX, *ROUTE, (4,)])
    supports = [
        (i, j, _cell_shape(env.frame(), i, j)[1])
        for i, y in enumerate(ROW_ANCHORS)
        for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] == 12
    ]
    print(
        "BASE", base_gain, avatar_cell(env.frame()), controls(env.frame()),
        "SUPPORTS", supports, flush=True,
    )
    results = []
    actions = [click_action(i, j) for i, j, _ in supports]
    for mask in range(1 << len(actions)):
        staged = tuple(a for k, a in enumerate(actions) if mask & (1 << k))
        for pattern in PATTERNS:
            node = env.clone()
            gain = base_gain + advance(node, staged)
            if node.terminal():
                continue
            gravity = tuple((6, 3, y) for y in controls(node.frame()))
            if len(gravity) != 1:
                continue
            path = (*staged, gravity[0], *pattern)
            gain += advance(node, (gravity[0], *pattern))
            if node.levels_completed > 6:
                print("WIN", path, "gain", gain, flush=True)
                return
            if node.terminal():
                continue
            results.append(
                (
                    gain,
                    len(controls(node.frame())),
                    avatar_cell(node.frame()),
                    mask,
                    pattern,
                    controls(node.frame()),
                )
            )
    results.sort(key=lambda x: (-x[0], -x[1], x[3], len(x[4])))
    print("TOP", *results[:40], sep="\n", flush=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
