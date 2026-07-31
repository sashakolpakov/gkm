import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift
from perception import connected_components, frame_delta


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def load_solver():
    spec = importlib.util.spec_from_file_location("clean_solve", "solve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def avatar(frame):
    blobs = connected_components(frame, colors=(9, 11), min_area=2)
    return [(blob.color, blob.bbox, blob.area) for blob in blobs]


def lattice(frame):
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
        12: "c", 14: "Y", 15: "f",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def result(base, node):
    return (
        node.levels_completed,
        node.terminal(),
        avatar(node.frame()),
        band_shift(base, node.frame()),
        frame_delta(base, node.frame())["count"],
    )


def probe(env):
    load_solver()(env)
    base = env.frame()
    print("START", env.levels_completed, env.actions, avatar(base), lattice(base))

    for action in (3, 4, 7):
        node = env.clone()
        node.step(action)
        print("KEY", action, result(base, node))

    changed = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            node = env.clone()
            node.step(6, x, y)
            delta = frame_delta(base, node.frame())
            if delta["count"] > 64:
                changed.append(
                    (i, j, _cell_shape(base, i, j), result(base, node))
                )
    print("CLICKS", changed)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
