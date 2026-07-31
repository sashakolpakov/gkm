import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components, frame_delta
from legs import _cell_shape, click_action


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ROWS = [3 + 6 * i for i in range(10)]
COLS = [15 + 6 * j for j in range(8)]


def avatar(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11), min_area=2)
    ]


def lattice(frame):
    palette = {
        3: "#",
        5: "#",
        8: "g",
        9: "A",
        10: ".",
        11: "a",
        12: "s",
        14: "Y",
        15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COLS)
        for y in ROWS
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base = env.frame()
    print("START", env.levels_completed, env.actions, avatar(base), lattice(base))
    for action in (action for action in env.actions if action != 6):
        node = env.clone()
        node.step(action)
        print(
            "KEY",
            action,
            node.levels_completed,
            node.terminal(),
            avatar(node.frame()),
            frame_delta(base, node.frame())["bbox"],
            lattice(node.frame()),
        )

    changed = []
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            node = env.clone()
            node.step(6, x, y)
            delta = frame_delta(base, node.frame())
            if delta["count"] > 2:
                changed.append(
                    (
                        (i, j),
                        int(base[y][x]),
                        delta["count"],
                        delta["bbox"],
                        avatar(node.frame()),
                    )
                )
    print("CLICKS", changed)
    for cell in ((2, 2), (4, 2), (4, 4), (6, 4), (8, 4)):
        node = env.clone()
        before = _cell_shape(node.frame(), *cell)
        node.step(*click_action(*cell))
        once = _cell_shape(node.frame(), *cell)
        node.step(*click_action(*cell))
        twice = _cell_shape(node.frame(), *cell)
        print("DOUBLE", cell, before, once, twice)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
