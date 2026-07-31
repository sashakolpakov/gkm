import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components, frame_delta
from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, run_actions


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    return [(b.bbox, b.area) for b in blobs]


def shapes(frame):
    return [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 15)
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    before = env.frame()
    print("START", avatar(before), shapes(before))
    tests = [
        ("control_" + str(x), [(6, x, 3)])
        for x in (3, 9, 15, 21, 27, 33, 39, 45, 51, 57)
    ]
    tests += [
        ("edge_" + str(y), [(6, 3, y)])
        for y in (3, 9, 15, 21, 27, 33, 39, 45, 51, 57)
    ]
    tests += [
        ("walk_" + str(n) + "_support8",
         [(4,)] * n + [(6, COL_ANCHORS[4], ROW_ANCHORS[8])])
        for n in range(5)
    ]
    tests += [
        ("walk2_support8_control_" + str(x),
         [(4,), (4,), (6, COL_ANCHORS[4], ROW_ANCHORS[8]), (6, x, 3)])
        for x in (3, 15, 21, 27, 33)
    ]
    for label, path in tests:
        node = env.clone()
        run_actions(node, path)
        after = node.frame()
        delta = frame_delta(before, after)
        print(label, "lvl", node.levels_completed, "term", node.terminal(),
              "avatar", avatar(after), "delta", delta["bbox"],
              "shapes", shapes(after))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
