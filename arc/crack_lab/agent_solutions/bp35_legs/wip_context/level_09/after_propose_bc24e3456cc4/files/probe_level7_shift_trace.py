import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def controls(frame):
    return tuple(y for y in ROW_ANCHORS if int(frame[y][3]) == 8)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw_route = json.load(stream)["staged_prefix_actions"]

    route = []
    for raw in raw_route:
        action = (raw,) if isinstance(raw, int) else tuple(raw)
        if len(action) == 3 and action[1] != 3:
            action = (action[0], action[1] + 12, action[2])
        route.append(action)

    node = env.clone()
    cumulative = 0
    for index, action in enumerate(route, 1):
        before = arr(node.frame()).copy()
        clicked = None
        if len(action) == 3 and action[1] != 3:
            i = ROW_ANCHORS.index(action[2])
            j = COL_ANCHORS.index(action[1])
            clicked = (i, j, _cell_shape(before, i, j))
        node.step(*action)
        after = node.frame()
        gain = 0 if node.terminal() else band_shift(before, after)
        cumulative += gain
        if clicked is not None:
            i, j, old_shape = clicked
            print(
                "OBJECT", index, action, (i, j), old_shape,
                _cell_shape(after, i, j), "gain", gain,
                "avatar", avatar_cell(after), "controls", controls(after),
            )
        elif gain:
            print(
                "RISE", index, action, gain, cumulative,
                avatar_cell(after), controls(after),
            )
        if node.terminal():
            print("TERMINAL", index, cumulative)
            break
    print(
        "FINAL", len(route), cumulative, node.levels_completed,
        node.terminal(), avatar_cell(node.frame()), controls(node.frame()),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
