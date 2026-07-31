"""Probe the verified post-column-1 level-6 frontier."""
import gkm_try as harness
import legs


PREFIX = [
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(5, 4),
    legs.click_action(4, 2),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    legs.click_action(5, 3), (legs.LEFT,),
    legs.click_action(5, 2), (legs.LEFT,),
    legs.click_action(5, 1), (legs.LEFT,),
    legs.click_action(5, 1),
    legs.click_action(8, 5),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(0, 6),
    (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
]


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def compact(node):
    frame = node.frame()
    return (
        node.levels_completed,
        node.terminal(),
        legs.moves_used(frame),
        avatar_cell(frame),
        tuple("".join(row) for row in legs.band_grid(frame)),
        tuple(
            (i, j, legs._cell_shape(frame, i, j))
            for i in range(legs.GRID_ROWS)
            for j in range(legs.GRID_COLS)
            if int(frame[legs.ROW_ANCHORS[i]][legs.COL_ANCHORS[j]])
            not in (3, 5, 9, 10, 11)
        ),
    )


def probe(env):
    harness.resumed_solve(env)
    root = env.clone()
    legs.run_actions(root, PREFIX)
    print("ROOT", compact(root))
    tests = {
        "left1": [(legs.LEFT,)],
        "left2": [(legs.LEFT,), (legs.LEFT,)],
        "right1": [(legs.RIGHT,)],
        "right2": [(legs.RIGHT,), (legs.RIGHT,)],
        "right3": [(legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,)],
        "right4": [
            (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
        ],
    }
    for col in (2, 3, 4, 5):
        tests[f"support{col}"] = [legs.click_action(5, col)]
    for name, path in tests.items():
        node = root.clone()
        legs.run_actions(node, path)
        print("TEST", name, path, compact(node))


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
