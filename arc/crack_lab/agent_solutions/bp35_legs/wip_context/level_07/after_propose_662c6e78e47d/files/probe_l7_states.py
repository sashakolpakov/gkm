"""Object-level snapshots at selected level-7 transition points."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, run_actions
from perception import color_counts, connected_components


R = (4,)
G = (6, 3, 3)


def click(i, j):
    return 6, COL_ANCHORS[j], ROW_ANCHORS[i]


def snapshot(root, name, route):
    node = root.clone()
    run_actions(node, route)
    frame = node.frame()
    actors = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(0, 7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    ]
    small_centers = [
        (x, y)
        for y in range(1, 62)
        for x in range(1, 63)
        if int(frame[y][x]) == 12
        and all(
            int(frame[y + dy][x + dx]) == 12
            for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        )
    ]
    print(
        name,
        {
            "steps": len(route),
            "level_delta": int(node.levels_completed - root.levels_completed),
            "terminal": bool(node.terminal()),
            "colors": color_counts(frame),
            "actors": actors,
            "cross_centers": small_centers,
        },
    )


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    routes = {
        "entry": [],
        "gravity": [G],
        "gravity_left": [G, (3,)],
        "gravity_right": [G, R],
        "bridge_drop": [R, R, R, click(8, 4), G],
        "ledge": [R, R, R, click(8, 4), G, R],
        "shaft": [R, R, R, click(8, 4), G, R, R],
        "upper_left_drop": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
        ],
        "hazard_underpass_drop": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
            (6, 27, 23),
            G,
            (3,),
            (3,),
            R,
            click(8, 1),
            G,
        ],
        "zero_obstacle": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
            (6, 27, 23),
            G,
            (3,),
            (3,),
            R,
            click(8, 1),
            (6, 3, 21),
            R,
        ],
        "deep_drop": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
            (6, 27, 23),
            G,
            (3,),
            (3,),
            R,
            click(9, 1),
            (6, 3, 21),
        ],
        "between_hazards": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
            (6, 27, 23),
            G,
            (3,),
            (3,),
            R,
            click(9, 1),
            (6, 3, 21),
            (6, 21, 23),
            (6, 27, 23),
            (6, 3, 5),
            R,
            (6, 33, 39),
            R,
        ],
        "left_descent": [
            R,
            R,
            R,
            click(8, 4),
            G,
            R,
            G,
            R,
            (3,),
            (3,),
            (3,),
            (3,),
            click(9, 2),
            G,
            (6, 27, 23),
            G,
            (3,),
            (3,),
            (6, 3, 21),
        ],
    }
    for name, route in routes.items():
        snapshot(env, name, route)


arena.run_program("bp35", probe)
