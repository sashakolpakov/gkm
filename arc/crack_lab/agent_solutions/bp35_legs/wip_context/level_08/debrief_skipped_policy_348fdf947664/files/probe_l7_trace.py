"""Compact transition traces for level-7 movement/support hypotheses."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import BAND, COL_ANCHORS, ROW_ANCHORS, band_shift


LEFT = (3,)
RIGHT = (4,)


def click(i, j):
    return 6, COL_ANCHORS[j], ROW_ANCHORS[i]


def control(i):
    return 6, 3, ROW_ANCHORS[i]


def cell_shape(frame, i, j):
    color = int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]])
    r0, c0 = BAND * i, 13 + BAND * j
    area = sum(
        int(frame[r][c]) == color
        for r in range(r0, r0 + BAND)
        for c in range(c0, c0 + BAND)
    )
    return color, area


def symbol(frame, i, j):
    color, area = cell_shape(frame, i, j)
    if color in (3, 5):
        return "#"
    if color == 10:
        return "."
    if color in (9, 11):
        return "A"
    if color == 12:
        return "X" if area > 5 else "x"
    if color == 14:
        return "Y" if area > 5 else "y"
    if color == 15:
        return "P"
    return str(color)[-1]


def grid(frame):
    return ["".join(symbol(frame, i, j) for j in range(8)) for i in range(10)]


def summary(frame):
    board = grid(frame)
    avatar = next(
        ((i, row.index("A")) for i, row in enumerate(board) if "A" in row),
        None,
    )
    return {
        "avatar": avatar,
        "small": sum(row.count("x") + row.count("y") for row in board),
        "large": sum(row.count("X") + row.count("Y") for row in board),
        "prizes": [
            (i, j)
            for i, row in enumerate(board)
            for j, value in enumerate(row)
            if value == "P"
        ],
    }


def signed_shift(before, after):
    """Signed whole-band camera shift: positive means the world slid down."""
    old = [tuple(int(value) for value in before[row][6:61]) for row in range(63)]
    new = [tuple(int(value) for value in after[row][6:61]) for row in range(63)]
    scores = []
    for bands in range(-10, 11):
        offset = BAND * bands
        if offset >= 0:
            hits = sum(
                old[row] == new[row + offset]
                for row in range(63 - offset)
            )
        else:
            hits = sum(
                old[row - offset] == new[row]
                for row in range(63 + offset)
            )
        scores.append((hits, bands))
    return max(scores)


def action_name(action):
    if len(action) == 1:
        return "L" if action[0] == 3 else "R"
    kind = action[0]
    x, y = action[1:]
    if x == 3:
        suffix = str(ROW_ANCHORS.index(y)) if y in ROW_ANCHORS else f"@{y}"
        return f"G{kind}:{suffix}"
    if x not in COL_ANCHORS or y not in ROW_ANCHORS:
        return f"C{kind}@{x},{y}"
    return f"C{kind}:{ROW_ANCHORS.index(y)}{COL_ANCHORS.index(x)}"


def trace(root, name, route):
    node = root.clone()
    print("TRACE", name, "start", summary(node.frame()))
    for index, action in enumerate(route, 1):
        before = node.frame()
        node.step(*action)
        hits, shift = signed_shift(before, node.frame())
        print(
            index,
            action_name(action),
            {
                **summary(node.frame()),
                "rise": band_shift(before, node.frame()),
                "signed": (shift, hits),
                "level_delta": int(node.levels_completed - root.levels_completed),
                "terminal": bool(node.terminal()),
            },
        )
        if node.terminal() or node.levels_completed > root.levels_completed:
            break
    print("ENDGRID", name, "/".join(grid(node.frame())))


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    traces = {
        "target_action6": [control(0), (6, 21, 52)],
        "target_action7": [control(0), (7, 21, 52)],
        "down_action7_empty": [control(0), (7, 15, 3)],
        "down_action7_wall": [control(0), (7, 21, 3)],
        "action7_control": [(7, 3, 3), (7, 3, 3)],
        "mixed_controls": [(6, 3, 3), (7, 3, 3)],
        "support_6_7": [click(6, 4), (7, COL_ANCHORS[4], ROW_ANCHORS[6])],
        "action7_ceiling": [(7, COL_ANCHORS[1], ROW_ANCHORS[5])],
        "action7_right_wall": [RIGHT, RIGHT, RIGHT, (7, COL_ANCHORS[5], ROW_ANCHORS[6])],
        "action7_expanded_right": [
            click(6, 4),
            RIGHT,
            RIGHT,
            (7, COL_ANCHORS[4], ROW_ANCHORS[6]),
        ],
        "action7_below": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            (7, 39, 35),
        ],
        "left": [LEFT] * 6,
        "right": [RIGHT] * 8,
        "expand_near": [click(6, 4)] + [RIGHT] * 8,
        "expand_below": [click(8, 4)] + [RIGHT] * 8,
        "expand_both": [click(6, 4), click(8, 4)] + [RIGHT] * 8,
        "stand_then_expand": [RIGHT] * 3 + [click(6, 4), LEFT, RIGHT],
        "stand_expand_below": [RIGHT] * 3 + [click(8, 4), LEFT, RIGHT],
        "beside_prizes": [click(4, 2), click(4, 4), LEFT, RIGHT],
        "bridge_prizes": [click(4, 2), click(4, 3), click(4, 4), LEFT, RIGHT],
        "repeat_support": [click(6, 4), click(6, 4), click(6, 4)],
        "gravity_twice": [control(0), control(0)],
        "gravity_right": [control(0)] + [RIGHT] * 8,
        "gravity_left": [control(0)] + [LEFT] * 6,
        "gravity_right_flip": [control(0)] + [RIGHT] * 4 + [control(0)],
        "lower_bridge": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            RIGHT,
        ],
        "small_bridge": [
            RIGHT,
            RIGHT,
            RIGHT,
            control(0),
            RIGHT,
            control(0),
            RIGHT,
        ],
        "candidate_goal": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            RIGHT,
            control(0),
            LEFT,
            control(0),
        ],
        "upper_entry": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
        ],
        "upper_right": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            RIGHT,
        ],
        "upper_right_flip": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            RIGHT,
            control(0),
            LEFT,
        ],
        "upper_left": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
        ],
        "right_exit": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            click(9, 5),
            control(0),
            (6, 45, 23),
            control(0),
            RIGHT,
            RIGHT,
        ],
        "left_support": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            LEFT,
        ],
        "left_support_turn": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            LEFT,
            control(0),
            LEFT,
        ],
        "left_double_support": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
        ],
        "left_shaft_descent": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            (6, 3, 21),
            RIGHT,
        ],
        "action7_hazard": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            (7, 27, 39),
            RIGHT,
        ],
        "first_hazard": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(6, 2),
            RIGHT,
        ],
        "hazard_underpass_drop": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(8, 1),
            control(0),
        ],
        "hazard_underpass_valid": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(8, 1),
            (6, 3, 21),
        ],
        "hazard_cycle": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(8, 1),
            (6, 3, 21),
            (6, 3, 11),
        ],
        "underpass_move": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(8, 1),
            (6, 3, 21),
            RIGHT,
            RIGHT,
        ],
        "deep_underpass": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(9, 1),
            (6, 3, 21),
            RIGHT,
            RIGHT,
            RIGHT,
        ],
        "weave_first_pair": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(9, 1),
            (6, 3, 21),
            (6, 21, 23),
            (6, 27, 23),
            (6, 3, 5),
            RIGHT,
            (6, 33, 39),
            RIGHT,
        ],
        "test_lower_marker": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(9, 1),
            (6, 3, 21),
            (6, 21, 23),
            (6, 27, 23),
            (6, 3, 5),
            RIGHT,
            (6, 33, 39),
            RIGHT,
            (6, 3, 39),
        ],
        "land_between_lower": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(9, 1),
            (6, 3, 21),
            (6, 21, 23),
            (6, 27, 23),
            (6, 3, 5),
            RIGHT,
            (6, 33, 39),
            RIGHT,
            (6, 33, 51),
            (6, 3, 39),
        ],
        "between_second_pair": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(9, 1),
            (6, 3, 21),
            (6, 21, 23),
            (6, 27, 23),
            (6, 3, 5),
            RIGHT,
            (6, 33, 39),
            RIGHT,
            (6, 33, 51),
            (6, 3, 39),
            click(4, 4),
            RIGHT,
            (7, 33, 27),
        ],
        "free_underpass": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            (6, 3, 21),
            RIGHT,
        ],
        "zero_overpass": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            click(9, 2),
            control(0),
            (6, 27, 23),
            control(0),
            LEFT,
            LEFT,
            RIGHT,
            click(8, 1),
            (6, 3, 21),
            RIGHT,
            (6, 3, 11),
            RIGHT,
            RIGHT,
            RIGHT,
        ],
        "left_target": [
            RIGHT,
            RIGHT,
            RIGHT,
            click(8, 4),
            control(0),
            RIGHT,
            control(0),
            RIGHT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            control(0),
            RIGHT,
        ],
    }
    selected = {
        name for name in os.environ.get("TRACE_NAMES", "").split(",") if name
    }
    for name, route in traces.items():
        if selected and name not in selected:
            continue
        trace(env, name, route)
    if not selected:
        for row in range(10):
            trace(env, f"control_{row}", [control(row)])


arena.run_program("bp35", probe)
