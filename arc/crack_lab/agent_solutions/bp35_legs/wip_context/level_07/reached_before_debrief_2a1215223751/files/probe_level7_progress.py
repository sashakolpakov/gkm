import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
    run_actions,
)
from perception import arr
from perception import connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    return blobs[0].bbox if blobs else None


def grid(frame):
    palette = {3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
               12: "c", 14: "Y", 15: "f"}
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def thin_supports(frame):
    return [
        (i, j) for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def unknowns(frame):
    known = {3, 5, 8, 9, 10, 11, 12, 14, 15}
    return [
        (i, j, int(frame[y][x]))
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) not in known
    ]


def edge_controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    start = arr(env.frame()).copy()
    gravity = (6, 3, 3)
    aligned = [(4,), (4,), (4,), click_action(8, 4), gravity]
    opener = aligned + [(4,), gravity, (4,)]
    room5 = (
        opener + [(3,), (3,), click_action(8, 3), gravity, (3,),
                  (6, 3, 9), (3,),
                  (4,), (4,), click_action(8, 2), (6, 3, 9),
                  (3,), (3,), (6, 3, 15), (3,), (3,)]
    )
    room5_inverted = room5 + [(4,), (6, 3, 21), (4,)]
    room6 = (
        room5 + [(4,), click_action(7, 2), (6, 3, 21), (4,),
                 (6, 3, 15)]
    )
    room7 = room6 + [(3,)]
    routes = [
        ("g", [gravity]),
        ("gg", [gravity, gravity]),
        ("aligned", aligned),
        ("opener", opener),
        ("room2", opener + [(3,)]),
        ("room2_left", opener + [(3,), (3,), (3,), (3,)]),
        (
            "aligned2",
            opener + [(3,), (3,), (3,), (3,), click_action(8, 2), gravity],
        ),
        (
            "room3",
            opener + [(3,), (3,), click_action(8, 3), gravity, (3,),
                      (6, 3, 9)],
        ),
        (
            "room4",
            opener + [(3,), (3,), click_action(8, 3), gravity, (3,),
                      (6, 3, 9), (3,)],
        ),
        (
            "room5",
            room5,
        ),
        ("room5_R", room5 + [(4,)]),
        ("room5_RR", room5 + [(4,), (4,)]),
        ("room5_RRR", room5 + [(4,), (4,), (4,)]),
        ("room5_R_f", room5 + [(4,), click_action(6, 2)]),
        ("room5_R_f_R", room5 + [(4,), click_action(6, 2), (4,)]),
        ("room5_R_f_RR", room5 + [(4,), click_action(6, 2), (4,), (4,)]),
        (
            "room5_stage",
            room5 + [(4,), click_action(7, 2), (6, 3, 21), (4,),
                     (6, 3, 15)],
        ),
        (
            "room5_stage_R",
            room5 + [(4,), click_action(7, 2), (6, 3, 21), (4,),
                     (6, 3, 15), (4,)],
        ),
        ("room5_c6", room5 + [click_action(6, 1)]),
        ("room5_c7", room5 + [click_action(7, 1)]),
        ("room5_c8", room5 + [click_action(8, 1)]),
        ("room5_g", room5 + [(6, 3, 21)]),
        ("room6", room6),
        ("room7", room7),
        ("room7_g", room7 + [(6, 3, 27)]),
        ("room7_g_R", room7 + [(6, 3, 27), (4,)]),
        ("room7_g_R_g", room7 + [(6, 3, 27), (4,), (6, 3, 21)]),
        (
            "room7_g_R_g_R",
            room7 + [(6, 3, 27), (4,), (6, 3, 21), (4,)],
        ),
        ("room5_g_c0", room5 + [(6, 3, 21), click_action(4, 1)]),
        ("room5_g_c0_R", room5 + [(6, 3, 21), click_action(4, 1), (4,)]),
        ("room5_R_g", room5 + [(4,), (6, 3, 21)]),
        ("room5_R_g_R", room5 + [(4,), (6, 3, 21), (4,)]),
        ("room5_R_g_RR", room5 + [(4,), (6, 3, 21), (4,), (4,)]),
        (
            "room5_R_g_R_c_g",
            room5 + [(4,), (6, 3, 21), (4,), click_action(3, 3),
                     (6, 3, 15)],
        ),
        (
            "room5_R_g_R_c_g_R",
            room5 + [(4,), (6, 3, 21), (4,), click_action(3, 3),
                     (6, 3, 15), (4,)],
        ),
        (
            "room5_R_g_R_c_g_L",
            room5 + [(4,), (6, 3, 21), (4,), click_action(3, 3),
                     (6, 3, 15), (3,)],
        ),
    ]
    routes += [
        ("room5_g_R" + str(n), room5 + [(6, 3, 21)] + [(4,)] * n)
        for n in range(1, 8)
    ]
    routes += [
        (
            "room5_inv_" + str(i) + str(j) + "_g",
            room5_inverted + [click_action(i, j), (6, 3, 15)],
        )
        for i, j in ((2, 4), (3, 3), (8, 2), (8, 3))
    ]
    routes.append(("room5_inv_g", room5_inverted + [(6, 3, 15)]))
    for label, route in routes:
        node = env.clone()
        run_actions(node, route)
        print(
            label,
            "depth", len(route),
            "shift", band_shift(start, node.frame()),
            "avatar", avatar(node.frame()),
            "terminal", node.terminal(),
            "levels", node.levels_completed,
        )
        if (
            label in ("room2", "room3", "room4", "room6")
            or label.startswith(("room5", "room7"))
        ):
            print(
                "STATE", label, grid(node.frame()), thin_supports(node.frame()),
                unknowns(node.frame()), edge_controls(node.frame()),
            )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
