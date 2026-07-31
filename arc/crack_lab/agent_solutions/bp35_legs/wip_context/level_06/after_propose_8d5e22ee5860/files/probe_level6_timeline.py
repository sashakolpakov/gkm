"""Compact transition timeline for the reproduced level-6 prefix."""
import gkm_try as harness
import legs


PREFIX = [
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(5, 4),
    (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
    legs.click_action(4, 2),
    legs.click_action(7, 0),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    legs.click_action(5, 3), (legs.LEFT,),
    legs.click_action(5, 2), (legs.LEFT,),
    legs.click_action(5, 2),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
]


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def summary(node):
    frame = node.frame()
    specials = tuple(
        (i, j, legs._cell_shape(frame, i, j))
        for i in range(legs.GRID_ROWS)
        for j in range(legs.GRID_COLS)
        if int(frame[legs.ROW_ANCHORS[i]][legs.COL_ANCHORS[j]])
        not in (3, 5, 9, 10, 11, 15)
    )
    return (
        legs.moves_used(frame),
        avatar_cell(frame),
        tuple("".join(row) for row in legs.band_grid(frame)),
        specials,
    )


def probe(env):
    harness.resumed_solve(env)
    node = env.clone()
    before = node.frame()
    print("AT", 0, None, 0, summary(node))
    for k, action in enumerate(PREFIX, 1):
        node.step(*action)
        after = node.frame()
        print(
            "AT", k, action, legs.band_shift(before, after),
            summary(node), "terminal", node.terminal(),
        )
        if node.terminal():
            break
        before = after
    variants = {
        "with_color7": PREFIX,
        "omit_color7": PREFIX[:10] + PREFIX[11:],
        "wall_noop": (
            PREFIX[:10] + [legs.click_action(0, 0)] + PREFIX[11:]
        ),
    }
    for name, path in variants.items():
        child = env.clone()
        legs.run_actions(child, path)
        print("VARIANT", name, summary(child), "terminal", child.terminal())


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
