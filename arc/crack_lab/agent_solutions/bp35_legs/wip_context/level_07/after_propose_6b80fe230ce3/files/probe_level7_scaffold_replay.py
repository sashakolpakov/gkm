import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, run_actions
from perception import connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


STAGED_PREFIX = [
    (6, 15, 15),
    (6, 15, 27),
    (6, 27, 27),
    (6, 21, 9),
    (4,), (4,), (4,),
    (6, 27, 51),
    (6, 3, 3),
    (4,),
    (6, 3, 3),
    (4,),
    (3,), (3,),
    (6, 21, 51),
    (6, 3, 3),
    (3,),
    (6, 3, 9),
    (6, 39, 27),
    (3,),
    (6, 33, 39),
    (6, 15, 33),
    (6, 33, 57),
    (6, 3, 33),
    (4,),
    (6, 3, 45),
    (4,), (4,), (3,),
    (6, 3, 9),
    (3,),
    (6, 3, 51),
    (3,),
    (6, 3, 15),
    (4,),
    (6, 3, 51),
    (3,),
    (6, 3, 21),
    (4,),
    (6, 3, 57),
    (6, 27, 33),
    (4,),
    (6, 3, 57),
    (3,),
    (6, 3, 15),
    (3,),
    (6, 3, 27),
    (4,),
    (6, 3, 39),
    (3,),
    (6, 3, 9),
    (6, 3, 45),
    (3,), (3,),
]


def summary(node):
    frame = node.frame()
    avatar = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11), min_area=2)
    ]
    switches = [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "a",
        12: "s", 14: "Y", 15: "h",
    }
    lattice = "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )
    return node.levels_completed, node.terminal(), avatar, switches, lattice


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    shifted = [
        (action[0], action[1] + 12, action[2])
        if len(action) == 3 and action[1] != 3 else action
        for action in STAGED_PREFIX
    ]
    for label, prefix in (("RAW", STAGED_PREFIX), ("SHIFTED_X", shifted)):
        node = env.clone()
        run_actions(node, prefix + [(3,), (3,), (3,), (3,)])
        print(label, "BEFORE_REWARD", len(prefix) + 4, summary(node))
        switches = [y for y in ROW_ANCHORS if int(node.frame()[y][3]) == 8]
        if switches:
            node.step(6, 3, max(switches))
        print(label, "AFTER_REWARD", len(prefix) + 5, summary(node))

    stage = env.clone()
    run_actions(stage, shifted)
    print("SHIFTED_STAGE", len(shifted), summary(stage))
    candidates = [(3,), (4,), (6, 9, 3)]
    frame = stage.frame()
    candidates += [(6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8]
    candidates += [
        (6, COL_ANCHORS[j], ROW_ANCHORS[i])
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14, 15)
    ]
    for candidate in candidates:
        node = stage.clone()
        run_actions(node, [candidate, (3,), (3,), (3,), (3,)])
        switches = [y for y in ROW_ANCHORS if int(node.frame()[y][3]) == 8]
        if switches:
            node.step(6, 3, max(switches))
        if node.levels_completed > 6:
            print("ONE_ACTION_WIN", candidate, summary(node))

    final = stage.clone()
    run_actions(final, [(3,), (3,), (3,), (3,)])
    switches = [y for y in ROW_ANCHORS if int(final.frame()[y][3]) == 8]
    if switches:
        final.step(6, 3, max(switches))
    frame = final.frame()
    after_candidates = [(3,), (4,), (6, 9, 3)]
    after_candidates += [
        (6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8
    ]
    after_candidates += [
        (6, COL_ANCHORS[j], ROW_ANCHORS[i])
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14, 15)
    ]
    for candidate in after_candidates:
        node = final.clone()
        node.step(*candidate)
        if node.levels_completed > 6:
            print("APPENDED_ACTION_WIN", candidate, summary(node))

    pre_final = stage.clone()
    run_actions(pre_final, [(3,), (3,), (3,), (3,)])
    for y in [row for row in ROW_ANCHORS if int(pre_final.frame()[row][3]) == 8]:
        branch = pre_final.clone()
        branch.step(6, 3, y)
        print("FINAL_SWITCH_BRANCH", y, summary(branch))
        frame = branch.frame()
        branch_actions = [(3,), (4,), (6, 9, 3)]
        branch_actions += [
            (6, 3, row)
            for row in ROW_ANCHORS
            if int(frame[row][3]) == 8
        ]
        branch_actions += [
            (6, COL_ANCHORS[j], ROW_ANCHORS[i])
            for i in range(10)
            for j in range(8)
            if _cell_shape(frame, i, j)[0] in (12, 14, 15)
        ]
        for action in branch_actions:
            node = branch.clone()
            node.step(*action)
            if node.levels_completed > 6:
                print("FINAL_BRANCH_WIN", y, action, summary(node))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
