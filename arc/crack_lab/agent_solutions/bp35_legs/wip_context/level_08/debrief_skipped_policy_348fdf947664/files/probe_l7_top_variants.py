"""Isolated bridge macros from the replay-stable level-7 top room."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, click_action
from perception import connected_components
from probe_level7_reward_recovery import PREFIX, SUFFIX


L, R = (3,), (4,)
TOP_ROUTE = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    (3,), (6, 3, 9), (4,), (6, 3, 39),
    L, L, L,
]
DROP = (6, 3, 27)
NEAR = click_action(4, 2)
FAR = click_action(4, 4)
UPPER_NEAR = click_action(2, 2)
UPPER_FAR = click_action(2, 3)
BRIDGE72 = [
    DROP, (7,), click_action(7, 2), R, (6, 3, 21),
    R, R, R, R,
]
TOP_LEFT = [DROP, (7,), R]
NEXT = [*BRIDGE72, (6, 3, 0)]
NEXT_CROSS = [*NEXT, click_action(6, 3), R]
T2_STAGE = [
    *NEXT,
    click_action(6, 5),
    click_action(6, 3),
    R,
]
STAGED_NEXT = [*BRIDGE72, click_action(3, 3), (6, 3, 0)]
PAIR24_STAGE = [
    *BRIDGE72, click_action(3, 3), click_action(2, 4),
    (6, 3, 0), R,
]
RETURN = [*BRIDGE72, L, L, L, L]

VARIANTS = {
    "drop_right": [DROP, R],
    "near_right": [DROP, NEAR, R],
    "near_rights": [DROP, NEAR, R, R, R, R],
    "far_right": [DROP, FAR, R],
    "both_rights": [DROP, NEAR, FAR, R, R, R, R],
    "upper_rights": [DROP, UPPER_NEAR, UPPER_FAR, R, R, R, R],
    "near_toggle_rights": [DROP, NEAR, (7,), R, R, R, R],
    "drop_7": [DROP, (7,)],
    "drop_7_rights": [DROP, (7,), R, R, R, R, R, R],
    "drop_7_r_7": [DROP, (7,), R, (7,)],
    "drop_7_rr_7": [DROP, (7,), R, R, (7,)],
    "drop_7_rrr_7": [DROP, (7,), R, R, R, (7,)],
    "bridge61": [DROP, (7,), click_action(6, 1), R, (6, 3, 21),
                 R, R, R, R],
    "bridge71": [DROP, (7,), click_action(7, 1), R, (6, 3, 21),
                 R, R, R, R],
    "bridge72": BRIDGE72,
    "bridge74": [DROP, (7,), click_action(7, 4), R, (6, 3, 21),
                 R, R, R, R],
    "bridge81": [DROP, (7,), click_action(8, 1), R, (6, 3, 21),
                 R, R, R, R],
    "bridge83": [DROP, (7,), click_action(8, 3), R, (6, 3, 21),
                 R, R, R, R],
    "bridge91": [DROP, (7,), click_action(9, 1), R, (6, 3, 21),
                 R, R, R, R],
    "bridge72_up": [*BRIDGE72, (7,)],
    "bridge72_up_rights": [*BRIDGE72, (7,), R, R, R, R, R],
    "bridge72_77": [*BRIDGE72, (7,), (7,)],
    "bridge72_777": [*BRIDGE72, (7,), (7,), (7,)],
    "bridge72_77_rights": [*BRIDGE72, (7,), (7,), R, R, R, R, R],
    "bridge72_777_rights": [*BRIDGE72, (7,), (7,), (7,),
                            R, R, R, R, R],
    "bridge72_7_left": [*BRIDGE72, (7,), L],
    "bridge72_7_lefts": [*BRIDGE72, (7,), L, L, L, L],
    "bridge72_left_7": [*BRIDGE72, L, (7,)],
    "bridge72_left_7_rights": [*BRIDGE72, L, (7,), R, R, R, R],
    "gap22": [*BRIDGE72, click_action(2, 2), R, R, R, R],
    "gap24": [*BRIDGE72, click_action(2, 4), R, R, R, R],
    "gap33": [*BRIDGE72, click_action(3, 3), R, R, R, R],
    "gap73": [*BRIDGE72, click_action(7, 3), R, R, R, R],
    "gap74": [*BRIDGE72, click_action(7, 4), R, R, R, R],
    "gap82": [*BRIDGE72, click_action(8, 2), R, R, R, R],
    "gap83": [*BRIDGE72, click_action(8, 3), R, R, R, R],
    "hazard6": [*TOP_LEFT, (6, 27, 39), R, R, R],
    "hazard6_7": [*TOP_LEFT, (6, 27, 39), (7,), R, R, R],
    "hazard6_77": [*TOP_LEFT, (6, 27, 39), (7,), (7,), R, R, R],
    "hazard66": [*TOP_LEFT, (6, 27, 39), (6, 27, 39), R, R, R],
    "hazard7_coord": [*TOP_LEFT, (7, 27, 39), R, R, R],
    "clear22_up": [*BRIDGE72, click_action(2, 2), (7,)],
    "clear22_up_rights": [*BRIDGE72, click_action(2, 2), (7,),
                          R, R, R, R],
    "clear22_select_up": [*BRIDGE72, click_action(2, 2),
                          (6, 3, 0), (7,)],
    "clear22_select_up_rights": [*BRIDGE72, click_action(2, 2),
                                 (6, 3, 0), (7,), R, R, R, R],
    "clear33_select_up": [*BRIDGE72, click_action(3, 3),
                          (6, 3, 0), (7,), R, R, R, R],
    "next_control": NEXT,
    "next_control_moves": [*BRIDGE72, (6, 3, 0), R, R, L, L],
    "stage73_control": [*BRIDGE72, click_action(7, 3), (6, 3, 0),
                        R, R, R],
    "stage74_control": [*BRIDGE72, click_action(7, 4), (6, 3, 0),
                        R, R, R],
    "stage82_control": [*BRIDGE72, click_action(8, 2), (6, 3, 0),
                        R, R, R],
    "stage83_control": [*BRIDGE72, click_action(8, 3), (6, 3, 0),
                        R, R, R],
    "next_r1": [*NEXT, R],
    "next_r2": [*NEXT, R, R],
    "next_r3": [*NEXT, R, R, R],
    "next_r4": [*NEXT, R, R, R, R],
    "next_r5": [*NEXT, R, R, R, R, R],
    "next_l1": [*NEXT, L],
    "next_l2": [*NEXT, L, L],
    "next_l3": [*NEXT, L, L, L],
    "next_clear3": [*NEXT, click_action(6, 3)],
    "next_clear3_r1": [*NEXT, click_action(6, 3), R],
    "next_clear3_r2": [*NEXT, click_action(6, 3), R, R],
    "next_clear3_r3": [*NEXT, click_action(6, 3), R, R, R],
    "next_clear3_r4": [*NEXT, click_action(6, 3), R, R, R, R],
    "next_cross_7": [*NEXT_CROSS, (7,)],
    "next_cross_7_r": [*NEXT_CROSS, (7,), R],
    "next_cross_7_l": [*NEXT_CROSS, (7,), L],
    "next_cross_77": [*NEXT_CROSS, (7,), (7,)],
    "next_cross_77_r": [*NEXT_CROSS, (7,), (7,), R],
    "preclear33_control": STAGED_NEXT,
    "preclear33_r": [*STAGED_NEXT, R],
    "preclear33_rr": [*STAGED_NEXT, R, R],
    "preclear33_r7": [*STAGED_NEXT, R, (7,)],
    "preclear33_r7r": [*STAGED_NEXT, R, (7,), R],
    "preclear33_r7rr": [*STAGED_NEXT, R, (7,), R, R],
    "preclear33_r77": [*STAGED_NEXT, R, (7,), (7,)],
    "preclear33_r777": [*STAGED_NEXT, R, (7,), (7,), (7,)],
    "preclear33_r7777": [*STAGED_NEXT, R, (7,), (7,), (7,), (7,)],
    "preclear33_r77_r": [*STAGED_NEXT, R, (7,), (7,), R],
    "preclear33_r777_r": [*STAGED_NEXT, R, (7,), (7,), (7,), R],
    "pair22": [*BRIDGE72, click_action(3, 3), click_action(2, 2),
               (6, 3, 0), R, (7,), R],
    "pair24": [*BRIDGE72, click_action(3, 3), click_action(2, 4),
               (6, 3, 0), R, (7,), R],
    "pair73": [*BRIDGE72, click_action(3, 3), click_action(7, 3),
               (6, 3, 0), R, (7,), R],
    "pair74": [*BRIDGE72, click_action(3, 3), click_action(7, 4),
               (6, 3, 0), R, (7,), R],
    "pair82": [*BRIDGE72, click_action(3, 3), click_action(8, 2),
               (6, 3, 0), R, (7,), R],
    "pair83": [*BRIDGE72, click_action(3, 3), click_action(8, 3),
               (6, 3, 0), R, (7,), R],
    "p24_7": [*PAIR24_STAGE, (7,)],
    "p24_77": [*PAIR24_STAGE, (7,), (7,)],
    "p24_777": [*PAIR24_STAGE, (7,), (7,), (7,)],
    "p24_7777": [*PAIR24_STAGE, (7,), (7,), (7,), (7,)],
    "p24_77_r": [*PAIR24_STAGE, (7,), (7,), R],
    "p24_777_r": [*PAIR24_STAGE, (7,), (7,), (7,), R],
    "p24_7777_r": [*PAIR24_STAGE, (7,), (7,), (7,), (7,), R],
    "rev24": [*BRIDGE72, click_action(2, 4), click_action(3, 3),
              (6, 3, 0), R, (7,), R],
    "rev73": [*BRIDGE72, click_action(7, 3), click_action(3, 3),
              (6, 3, 0), R, (7,), R],
    "rev74": [*BRIDGE72, click_action(7, 4), click_action(3, 3),
              (6, 3, 0), R, (7,), R],
    "rev82": [*BRIDGE72, click_action(8, 2), click_action(3, 3),
              (6, 3, 0), R, (7,), R],
    "rev83": [*BRIDGE72, click_action(8, 3), click_action(3, 3),
              (6, 3, 0), R, (7,), R],
    "return": RETURN,
    "return_r": [*RETURN, R],
    "return_7": [*RETURN, (7,)],
    "return_77": [*RETURN, (7,), (7,)],
    "return_7_r": [*RETURN, (7,), R],
    "return_77_r": [*RETURN, (7,), (7,), R],
    "return_777_r": [*RETURN, (7,), (7,), (7,), R],
    "return_l1": [*BRIDGE72, L],
    "return_l2": [*BRIDGE72, L, L],
    "return_l3": [*BRIDGE72, L, L, L],
    "return_l1_7": [*BRIDGE72, L, (7,)],
    "return_l2_7": [*BRIDGE72, L, L, (7,)],
    "return_l3_7": [*BRIDGE72, L, L, L, (7,)],
    "l1_control": [*BRIDGE72, L, (6, 3, 0)],
    "l1_control_l": [*BRIDGE72, L, (6, 3, 0), L],
    "l1_control_r": [*BRIDGE72, L, (6, 3, 0), R],
    "l1_control_7": [*BRIDGE72, L, (6, 3, 0), (7,)],
    "l1_control_7_l": [*BRIDGE72, L, (6, 3, 0), (7,), L],
    "l1_control_7_r": [*BRIDGE72, L, (6, 3, 0), (7,), R],
    "far5_6": [*NEXT_CROSS, click_action(6, 5)],
    "far5_6_7": [*NEXT_CROSS, click_action(6, 5), (7,)],
    "far5_6_7_r": [*NEXT_CROSS, click_action(6, 5), (7,), R],
    "far5_6_77": [*NEXT_CROSS, click_action(6, 5), (7,), (7,)],
    "far5_6_77_r": [*NEXT_CROSS, click_action(6, 5), (7,), (7,), R],
    "far5_6_control": [*NEXT_CROSS, click_action(6, 5),
                       (6, 3, 21), R],
    "t2_stage": T2_STAGE,
    "t2_stage_7": [*T2_STAGE, (7,)],
    "t2_stage_7_r": [*T2_STAGE, (7,), R],
    "t2_stage_77": [*T2_STAGE, (7,), (7,)],
    "t2_stage_77_r": [*T2_STAGE, (7,), (7,), R],
    "t2_reverse": [*NEXT, click_action(6, 3), click_action(6, 5),
                   R, (7,), R],
}

for _cell in (
    (2, 5), (3, 5), (4, 5), (5, 5), (6, 1), (6, 5),
    (7, 1), (7, 2), (7, 4), (7, 5), (8, 1), (8, 3),
    (8, 5), (9, 1),
):
    VARIANTS[f"t2_{_cell[0]}{_cell[1]}"] = [
        *NEXT,
        click_action(*_cell),
        click_action(6, 3),
        R,
        (7,),
        R,
    ]
    if _cell != (6, 3):
        VARIANTS[f"pre_t2_{_cell[0]}{_cell[1]}"] = [
            *STAGED_NEXT,
            click_action(*_cell),
            R,
            (7,),
            R,
        ]


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def target(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
        10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def controls(frame):
    return [
        round(blob.centroid[0])
        for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]


def support_shapes(frame):
    out = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            if int(frame[y][x]) != 12:
                continue
            r0, c0 = 6 * i, 13 + 6 * j
            area = sum(
                int(frame[r][c]) == 12
                for r in range(r0, min(63, r0 + 6))
                for c in range(c0, c0 + 6)
            )
            out.append((i, j, area))
    return out


def run_variant(name, suffix):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        all_actions = [*TOP_ROUTE, *suffix]
        for index, action in enumerate(all_actions, 1):
            env.step(*action)
            if name == "bridge72_up" and index > len(TOP_ROUTE):
                print(
                    "TOP_STEP", index - len(TOP_ROUTE), action,
                    env.terminal(), avatar(env.frame()), controls(env.frame()),
                    lattice(env.frame()), support_shapes(env.frame()),
                )
            if env.terminal() or env.levels_completed > 6:
                result["stopped"] = index
                break
        result.update(
            level=int(env.levels_completed),
            terminal=bool(env.terminal()),
            avatar=avatar(env.frame()),
            target=target(env.frame()),
            controls=controls(env.frame()),
            lattice=lattice(env.frame()),
        )

    levels, path, error = arena.run_program("bp35", probe)
    print(
        "TOP_VARIANT", name, len(suffix), result,
        "runner", (levels, len(path), error), flush=True,
    )


only = set(filter(None, os.environ.get("ONLY", "").split(",")))
for variant_name, variant_suffix in VARIANTS.items():
    if not only or variant_name in only:
        run_variant(variant_name, variant_suffix)
