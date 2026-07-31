"""Test every protected lane at the level-6 full support barrier."""
import gkm_try as harness
import legs


BARRIER = [
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(5, 4),
    legs.click_action(4, 2),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
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
            not in (3, 5, 9, 10, 11, 15)
        ),
    )


def protected_walk(target):
    path = []
    for col in range(3, target - 1, -1):
        path.extend([legs.click_action(5, col), (legs.LEFT,)])
    return path


def probe(env):
    harness.resumed_solve(env)
    root = env.clone()
    legs.run_actions(root, BARRIER)
    print("ROOT", compact(root))
    tests = {}
    for target in (3, 2, 1, 0):
        path = protected_walk(target)
        tests[f"to{target}"] = path
        tests[f"to{target}_activate"] = path + [
            legs.click_action(5, target)
        ]
    for name, path in tests.items():
        node = root.clone()
        legs.run_actions(node, path)
        print("TEST", name, path, compact(node))
    for lane in (1, 0):
        stage = root.clone()
        entry = protected_walk(lane) + [legs.click_action(5, lane)]
        legs.run_actions(stage, entry)
        branch_tests = {
            "left": [(legs.LEFT,)],
            "support0": [legs.click_action(8, 0)],
            "support1": [legs.click_action(8, 1)],
            "left_support0": [
                (legs.LEFT,), legs.click_action(8, 0),
            ],
            "aligned_support_then_right": [
                legs.click_action(8, lane), (legs.RIGHT,),
            ],
            "right1": [(legs.RIGHT,)],
            "right2": [(legs.RIGHT,), (legs.RIGHT,)],
            "right3": [(legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,)],
            "right4": [
                (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
            ],
            "right5": [
                (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
                (legs.RIGHT,),
            ],
        }
        for name, path in branch_tests.items():
            node = stage.clone()
            legs.run_actions(node, path)
            print("BRANCH", lane, name, path, compact(node))
        if lane == 1:
            configurable = (0, 1, 5, 6)
            forced = [
                (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
                legs.click_action(0, 6),
                (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
            ]
            for mask in range(1 << len(configurable)):
                setup = [
                    legs.click_action(8, col)
                    for bit, col in enumerate(configurable)
                    if mask & (1 << bit)
                ]
                node = stage.clone()
                legs.run_actions(node, setup + forced)
                print(
                    "CONFIG",
                    tuple(col for bit, col in enumerate(configurable)
                          if mask & (1 << bit)),
                    node.levels_completed,
                    node.terminal(),
                    legs.moves_used(node.frame()),
                    avatar_cell(node.frame()),
                    tuple("".join(row)
                          for row in legs.band_grid(node.frame())),
                )
                finish = forced + [
                    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
                    (legs.LEFT,),
                    (legs.LEFT,), (legs.LEFT,),
                    (legs.RIGHT,), (legs.RIGHT,),
                ]
                node = stage.clone()
                legs.run_actions(node, setup + finish)
                print(
                    "FINISH_CONFIG",
                    tuple(col for bit, col in enumerate(configurable)
                          if mask & (1 << bit)),
                    node.levels_completed,
                    node.terminal(),
                    legs.moves_used(node.frame()),
                    avatar_cell(node.frame()),
                    tuple("".join(row)
                          for row in legs.band_grid(node.frame())),
                )


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
