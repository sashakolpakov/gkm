import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import color_counts, connected_components
from legs import COL_ANCHORS, ROW_ANCHORS, click_action, run_actions


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def summary(node):
    frame = node.frame()
    blobs = connected_components(frame, min_area=2)
    interesting = [
        (b.color, b.bbox, b.area)
        for b in blobs if b.color not in (5, 10)
    ]
    palette = {3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
               12: "c", 14: "Y", 15: "f"}
    grid = "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )
    return (
        node.levels_completed, node.terminal(), color_counts(frame),
        interesting, grid,
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    support8 = click_action(8, 4)
    gravity = (6, 3, 3)
    stage = [support8, gravity]
    cross = stage + [(4,), (4,), click_action(4, 4), (4,), gravity]
    aligned = [(4,), (4,), (4,), support8, gravity]
    opener = aligned + [(4,), gravity, (4,)]
    room2 = opener + [(3,)]
    room2_cross = (
        room2 + [(3,), (3,), (3,), click_action(8, 2),
                 gravity, (3,), gravity, (3,)]
    )
    aligned2 = room2 + [(3,), (3,), (3,), click_action(8, 2), gravity]
    stage2 = aligned2 + [(4,)]
    tests = [
        ("g_g", [gravity, gravity]),
        ("g_R_g", [gravity, (4,), gravity]),
        ("g_RR_g", [gravity, (4,), (4,), gravity]),
        ("g_RRR_g", [gravity, (4,), (4,), (4,), gravity]),
        ("g_L_g", [gravity, (3,), gravity]),
        ("aligned", aligned),
        ("aligned_g", aligned + [gravity]),
        ("aligned_L", aligned + [(3,)]),
        ("aligned_L_g", aligned + [(3,), gravity]),
        ("aligned_R", aligned + [(4,)]),
        ("aligned_R_g", aligned + [(4,), gravity]),
        ("aligned_R_g_L", aligned + [(4,), gravity, (3,)]),
        ("aligned_R_g_R", aligned + [(4,), gravity, (4,)]),
        ("aligned_R_g_g", aligned + [(4,), gravity, gravity]),
        ("aligned_RR", aligned + [(4,), (4,)]),
        ("aligned_RRR", aligned + [(4,), (4,), (4,)]),
        ("aligned_RRRR", aligned + [(4,), (4,), (4,), (4,)]),
        ("aligned_RR_g", aligned + [(4,), (4,), gravity]),
        ("opener", opener),
        ("opener_R", opener + [(4,)]),
        ("opener_g", opener + [gravity]),
        ("opener_R_g", opener + [(4,), gravity]),
        ("opener_R_g_L", opener + [(4,), gravity, (3,)]),
        ("opener_R_g_LL", opener + [(4,), gravity, (3,), (3,)]),
        ("opener_R_g_LLL", opener + [(4,), gravity, (3,), (3,), (3,)]),
        ("opener_R_g_R", opener + [(4,), gravity, (4,)]),
        ("room2", room2),
        ("room2_L", room2 + [(3,)]),
        ("room2_LL", room2 + [(3,), (3,)]),
        ("room2_LLL", room2 + [(3,), (3,), (3,)]),
        ("room2_R", room2 + [(4,)]),
        ("room2_g", room2 + [gravity]),
        ("room2_cross", room2_cross),
        ("room2_cross_R", room2_cross + [(4,)]),
        ("room2_cross_L", room2_cross + [(3,)]),
        ("aligned2", aligned2),
        ("aligned2_L", aligned2 + [(3,)]),
        ("aligned2_LL", aligned2 + [(3,), (3,)]),
        ("aligned2_R", aligned2 + [(4,)]),
        ("aligned2_L_g", aligned2 + [(3,), gravity]),
        ("aligned2_L_g_L", aligned2 + [(3,), gravity, (3,)]),
        ("aligned2_L_g_R", aligned2 + [(3,), gravity, (4,)]),
        ("stage2", stage2),
        ("stage2_L", stage2 + [(3,)]),
        ("stage2_R", stage2 + [(4,)]),
        ("stage2_g", stage2 + [gravity]),
        ("stage2_g_L", stage2 + [gravity, (3,)]),
        ("stage2_g_R", stage2 + [gravity, (4,)]),
        ("stage", stage),
        ("stage_R", stage + [(4,)]),
        ("stage_L", stage + [(3,)]),
        ("stage_LL", stage + [(3,), (3,)]),
        ("stage_LLL", stage + [(3,), (3,), (3,)]),
        ("stage_RR", stage + [(4,), (4,)]),
        ("stage_RR_click44", stage + [(4,), (4,), click_action(4, 4)]),
        ("stage_RR_click44_LR", stage + [(4,), (4,), click_action(4, 4), (3,), (4,)]),
        ("stage_RR_click44_R", stage + [(4,), (4,), click_action(4, 4), (4,)]),
        ("stage_RR_click44_g", stage + [(4,), (4,), click_action(4, 4), gravity]),
        ("cross", cross),
        ("cross_L", cross + [(3,)]),
        ("cross_R", cross + [(4,)]),
        ("cross_g", cross + [gravity]),
        ("cross_click64", cross + [click_action(6, 4)]),
        ("cross_click64_g", cross + [click_action(6, 4), gravity]),
        ("cross_click84_g", cross + [click_action(8, 4), gravity]),
    ]
    for label, route in tests:
        node = env.clone()
        run_actions(node, route)
        print(label, summary(node))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
