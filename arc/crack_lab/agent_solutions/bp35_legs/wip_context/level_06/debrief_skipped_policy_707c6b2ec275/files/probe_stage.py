"""Inspect one verified level-6 prefix and its immediate affordances."""
import gkm_try as harness
import legs


PREFIX = [
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.RIGHT,),
    legs.click_action(5, 4),
    (legs.LEFT,),
    (legs.LEFT,),
    (legs.LEFT,),
    legs.click_action(4, 2),
    legs.click_action(7, 0),
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.LEFT,),
    (legs.LEFT,),
    (legs.RIGHT,),
    (legs.RIGHT,),
    (legs.LEFT,),
    (legs.LEFT,),
]


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def compact(node):
    frame = node.frame()
    specials = []
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            color = int(frame[y][x])
            if color not in (3, 5, 10):
                specials.append((i, j, legs._cell_shape(frame, i, j)))
    return (
        node.levels_completed,
        node.terminal(),
        legs.moves_used(frame),
        avatar_cell(frame),
        tuple("".join(row) for row in legs.band_grid(frame)),
        tuple(specials),
    )


def probe(env):
    harness.resumed_solve(env)
    root = env.clone()
    legs.run_actions(root, PREFIX)
    print("STAGE", compact(root))
    actions = [(legs.LEFT,), (legs.RIGHT,)]
    frame = root.frame()
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) not in (3, 5, 9, 10, 11):
                actions.append(legs.click_action(i, j))
    for action in actions:
        child = root.clone()
        child.step(*action)
        print("NEXT", action, compact(child))
    experiments = {
        "support_c2": [legs.click_action(5, 2)],
        "support_c1": [(legs.LEFT,), legs.click_action(5, 1)],
        "support_c3": [(legs.RIGHT,), legs.click_action(5, 3)],
        "left2": [(legs.LEFT,), (legs.LEFT,)],
        "right2": [(legs.RIGHT,), (legs.RIGHT,)],
    }
    for name, path in experiments.items():
        child = root.clone()
        legs.run_actions(child, path)
        print("SEQ", name, path, compact(child))


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
