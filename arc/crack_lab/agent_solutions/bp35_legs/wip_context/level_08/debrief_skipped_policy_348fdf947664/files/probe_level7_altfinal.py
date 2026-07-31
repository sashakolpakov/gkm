import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions
from perception import color_counts, connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ROUTE = [
    click_action(2, 2),
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
    (3,), (6, 3, 33), (4,), (6, 3, 33),
    (6, 3, 15), (3,), (6, 3, 45),
    (6, 3, 57), (4,), (6, 3, 51),
    (3,), (6, 3, 27), (4,), (6, 3, 57),
]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, ROUTE)
    frame = env.frame()
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "P",
        12: "c", 14: "Y", 15: "f",
    }
    grid = "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )
    shapes = [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] not in (3, 5, 10)
    ]
    objects = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
        if blob.color not in (3, 5, 10)
    ]
    print(
        "FINAL", env.levels_completed, env.terminal(), color_counts(frame),
        grid, shapes, objects,
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
