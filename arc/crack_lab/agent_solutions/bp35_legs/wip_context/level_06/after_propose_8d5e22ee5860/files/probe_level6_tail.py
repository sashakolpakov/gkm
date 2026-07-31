"""Compact clone probes after the reproduced level-6 stage."""
import gkm_try as harness
import legs


PREFIX = [
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(5, 4),
    legs.click_action(4, 2),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    legs.click_action(5, 3),
    (legs.LEFT,), (legs.LEFT,),
]

SAFE_LANE = []
NEXT_SHAFT = [(legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,)]
FLIP = [legs.click_action(0, 6)]


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def compact(node):
    frame = node.frame()
    changed = tuple(
        (i, j, legs._cell_shape(frame, i, j))
        for i in range(legs.GRID_ROWS)
        for j in range(legs.GRID_COLS)
        if int(frame[legs.ROW_ANCHORS[i]][legs.COL_ANCHORS[j]])
        not in (3, 5, 9, 10, 11)
    )
    return (
        node.levels_completed,
        node.terminal(),
        legs.moves_used(frame),
        avatar_cell(frame),
        tuple("".join(row) for row in legs.band_grid(frame)),
        changed,
    )


def probe(env):
    harness.resumed_solve(env)
    root = env.clone()
    legs.run_actions(root, PREFIX + SAFE_LANE + NEXT_SHAFT + FLIP)
    tests = {
        "right": [(legs.RIGHT,)],
        "left1": [(legs.LEFT,)],
        "left2": [(legs.LEFT,), (legs.LEFT,)],
        "left3": [(legs.LEFT,), (legs.LEFT,), (legs.LEFT,)],
        "left_right_left": [
            (legs.LEFT,), (legs.RIGHT,), (legs.LEFT,),
        ],
        "right_left_right": [
            (legs.RIGHT,), (legs.LEFT,), (legs.RIGHT,),
        ],
        "wall_clicks": [
            legs.click_action(0, 0),
            legs.click_action(0, 0),
            legs.click_action(0, 0),
        ],
        "hazard3": [legs.click_action(6, 3)],
        "hazard4": [legs.click_action(6, 4)],
        "support3": [legs.click_action(9, 3)],
        "support4": [legs.click_action(9, 4)],
        "support3_left3": [
            legs.click_action(9, 3),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "support4_left3": [
            legs.click_action(9, 4),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "supports_both_left3": [
            legs.click_action(9, 3), legs.click_action(9, 4),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "align3_support_left": [
            (legs.LEFT,), (legs.LEFT,),
            legs.click_action(9, 3), (legs.LEFT,),
        ],
        "align4_support_left2": [
            (legs.LEFT,), legs.click_action(9, 4),
            (legs.LEFT,), (legs.LEFT,),
        ],
        "align3_hazard": [
            (legs.LEFT,), (legs.LEFT,), legs.click_action(6, 3),
        ],
        "align3_hazard_left": [
            (legs.LEFT,), (legs.LEFT,), legs.click_action(6, 3),
            (legs.LEFT,),
        ],
        "align4_hazard_left2": [
            (legs.LEFT,), legs.click_action(6, 4),
            (legs.LEFT,), (legs.LEFT,),
        ],
        "hazard3_then_left3": [
            legs.click_action(6, 3),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "hazard4_then_left3": [
            legs.click_action(6, 4),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "hazard5_then_left3": [
            legs.click_action(6, 5),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "hazard6_then_left3": [
            legs.click_action(6, 6),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
        "all_hazards_then_left3": [
            legs.click_action(6, 3), legs.click_action(6, 4),
            legs.click_action(6, 5), legs.click_action(6, 6),
            (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
        ],
    }
    print("ROOT", compact(root))
    for name, path in tests.items():
        node = root.clone()
        legs.run_actions(node, path)
        print("TEST", name, path, compact(node))
    for action in ((7,), (7, legs.COL_ANCHORS[4], legs.ROW_ANCHORS[6])):
        node = root.clone()
        try:
            node.step(*action)
            result = compact(node)
        except Exception as exc:
            result = (type(exc).__name__, str(exc))
        print("ACTION7", action, result)


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
