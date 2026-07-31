"""Compact clean-room probes for the first unsolved bp35 level."""
import gkm_try as harness
import legs
import perception as p


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(frame, cell=4, min_area=4)
    ]


def summary(node):
    frame = node.frame()
    actors = [
        (b.color, b.bbox, b.area)
        for b in p.connected_components(frame, colors=(8, 9, 11, 12), min_area=1)
    ]
    shapes = tuple(
        (i, j, legs._cell_shape(frame, i, j))
        for i in range(legs.GRID_ROWS)
        for j in range(legs.GRID_COLS)
        if int(frame[legs.ROW_ANCHORS[i]][legs.COL_ANCHORS[j]])
        not in (3, 5, 10)
    )
    return (
        node.levels_completed,
        node.terminal(),
        legs.moves_used(frame),
        actors,
        shapes,
    )


def experiment(env, name, actions):
    node = env.clone()
    for action in actions:
        node.step(*action)
    print("SEQ", name, actions, summary(node))


def probe(env):
    harness.resumed_solve(env)
    print("LEVEL", env.levels_completed)
    print("ACTIONS", tuple(env.actions))
    print("COLORS", p.color_counts(env.frame()))
    print("OBJECTS", compact_objects(env.frame()))
    print("GRID", tuple("".join(row) for row in legs.band_grid(env.frame())))
    for action, delta in p.action_deltas(env, tuple(env.actions)).items():
        clone = env.clone()
        clone.step(action)
        print(
            "ACTION",
            action,
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "delta",
            {k: delta[k] for k in ("count", "bbox")},
            "objects",
            compact_objects(clone.frame()),
        )
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            color = int(env.frame()[y][x])
            if color not in (3, 5, 10):
                clone = env.clone()
                before = clone.frame()
                clone.step(6, x, y)
                delta = p.frame_delta(before, clone.frame())
                print(
                    "CLICK",
                    (i, j),
                    "color",
                    color,
                    "level",
                    clone.levels_completed,
                    "terminal",
                    clone.terminal(),
                    "delta",
                    {k: delta[k] for k in ("count", "bbox")},
                )
    left = (legs.LEFT,)
    right = (legs.RIGHT,)
    toggle = legs.click_action(5, 4)
    lower4 = legs.click_action(8, 4)
    lower5 = legs.click_action(8, 5)
    upper0 = legs.click_action(3, 0)
    upper1 = legs.click_action(3, 1)
    upper2 = legs.click_action(3, 2)
    experiments = {
        "lower4": [lower4],
        "lower5": [lower5],
        "upper0": [upper0],
        "near_toggle": [right, right, right, toggle],
        "lower4_toggle": [lower4, toggle],
        "lower5_toggle": [lower5, toggle],
        "lower_both_toggle": [lower4, lower5, toggle],
        "upper_all_toggle": [upper0, upper1, upper2, toggle],
        "all_supports_toggle": [
            upper0, upper1, upper2, lower4, lower5, toggle
        ],
        "c4_lower4_toggle": [right, right, right, lower4, toggle],
        "lower4_c4_toggle": [lower4, right, right, right, toggle],
        "c5_lower5_toggle": [right, right, right, right, lower5, toggle],
        "lower5_c5_toggle": [lower5, right, right, right, right, toggle],
    }
    for col in range(7):
        movement = [left] if col == 0 else [right] * (col - 1)
        experiments[f"toggle_c{col}"] = movement + [toggle]
    for name, actions in experiments.items():
        experiment(env, name, actions)
    plan = legs.gravity_room_search(
        env,
        max_states=2400,
        max_depth=80,
        interaction_radius=2,
        debug=False,
    )
    solved = env.clone()
    legs.run_actions(solved, plan)
    print(
        "GRAVITY_PLAN",
        len(plan),
        plan,
        "level",
        solved.levels_completed,
        "terminal",
        solved.terminal(),
    )


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
