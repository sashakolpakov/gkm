# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def prime_board(env, action=1):
    """Spend the board's initial inert input so subsequent controls take effect."""
    env.step(action)


def select_grid_cells_of_color(env, xs, ys, color, click_action=6):
    """Click the sampled cells of a coordinate grid that currently match color."""
    frame = env.frame()
    points = [
        (x, y)
        for y in ys
        for x in xs
        if int(frame[y][x]) == color
    ]
    for x, y in points:
        env.step(click_action, x, y)


def move_until_level_progress(env, action, max_steps):
    """Repeat a movement while bounded, stopping as soon as the level advances."""
    starting_level = env.levels_completed
    for _ in range(max_steps):
        if env.terminal() or env.levels_completed > starting_level:
            break
        env.step(action)


def probe_level_3_observations(env):
    """Temporary compact clone probes for level-3 mechanic discovery."""
    import perception as p

    def summary(node):
        blobs = [
            (b.color, b.area, b.bbox)
            for b in p.connected_components(node.frame(), min_area=4)
            if b.color != 5
        ]
        return node.levels_completed, p.color_counts(node.frame()), blobs

    print("L3 root", summary(env))
    board = p.arr(env.frame())
    rows = []
    for y in range(8, 44):
        runs = []
        start = 20
        for x in range(21, 61):
            if int(board[y, x]) != int(board[y, x - 1]):
                runs.append((int(board[y, start]), start, x - 1))
                start = x
        runs.append((int(board[y, start]), start, 60))
        if len(runs) > 1 or runs[0][0] != 5:
            rows.append((y, runs))
    print("L3 playfield_rows", rows)
    print(
        "L3 actions",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in p.action_deltas(env, actions=(1, 2, 3, 4)).items()
        },
    )

    selected = env.clone()
    for y in (50, 55, 60):
        for x in (25, 30, 35):
            clicked = env.clone()
            center = int(clicked.frame()[y][x])
            before_core = p.arr(clicked.frame())[:, :62].copy()
            clicked.step(6, x, y)
            core_delta = p.frame_delta(before_core, p.arr(clicked.frame())[:, :62])
            print(
                "L3 grid_cell",
                (x, y),
                center,
                (core_delta["count"], core_delta["bbox"]),
            )
    points = [
        (x, y)
        for y in (50, 55, 60)
        for x in (25, 30, 35)
        if int(selected.frame()[y][x]) == 0
    ]
    before = selected.frame()
    for index, (x, y) in enumerate(points, start=1):
        selected.step(6, x, y)
        partial_actions = p.action_deltas(selected, actions=(1, 2, 3, 4))
        print(
            "L3 partial_grid_actions",
            index,
            {
                action: (delta["count"], delta["bbox"])
                for action, delta in partial_actions.items()
            },
        )
    grid_delta = p.frame_delta(before, selected.frame())
    print(
        "L3 grid",
        points,
        (grid_delta["count"], grid_delta["bbox"]),
        summary(selected),
    )

    routes = {
        "push_once": [2, 2, 3, 3, 2],
        "push_twice": [2, 2, 3, 3, 2, 2],
        "stand_left": [2, 2, 2, 3, 3],
        "stand_left_push": [2, 2, 2, 3, 3, 2],
    }
    for root_name, root in (("raw", env), ("selected", selected)):
        for route_name, actions in routes.items():
            node = p.replay(root, actions)
            delta = p.frame_delta(root.frame(), node.frame())
            print(
                "L3 route",
                root_name,
                route_name,
                actions,
                (delta["count"], delta["bbox"]),
                summary(node),
            )
        path = p.bounded_bfs(
            root,
            p.level_goal(root.levels_completed),
            actions=(1, 2, 3, 4),
            key_fn=lambda node: p.arr(node.frame())[:, :62].tobytes(),
            max_states=5000,
            max_depth=64,
        )
        print("L3 bfs", root_name, path)

    contexts = {
        "right_gate": [4, 4, 4, 4],
        "bottom_gate": [3, 3, 2, 2],
        "top_tip": [1, 1, 1],
    }
    click_points = {
        "right_socket": (56, 24),
        "right_upper_bracket": (48, 22),
        "right_lower_bracket": (48, 25),
        "bottom_block": (28, 35),
        "bottom_shape": (24, 39),
    }
    key = lambda node: p.arr(node.frame())[:, :62].tobytes()
    for context_name, route in contexts.items():
        context = p.replay(selected, route)
        expected = {}
        for action in (1, 2, 3, 4):
            child = context.clone()
            child.step(action)
            expected[action] = (child.levels_completed, key(child))
        for point_name, point in click_points.items():
            clicked = context.clone()
            before_key = key(clicked)
            clicked.step(6, *point)
            effects = []
            if clicked.levels_completed != context.levels_completed or key(clicked) != before_key:
                effects.append(("direct", clicked.levels_completed))
            for action in (1, 2, 3, 4):
                child = clicked.clone()
                child.step(action)
                if (child.levels_completed, key(child)) != expected[action]:
                    effects.append((action, child.levels_completed))
            if effects:
                print(
                    "L3 contextual_click",
                    context_name,
                    route,
                    point_name,
                    point,
                    effects,
                )
        used = context.clone()
        before_key = key(used)
        used.step(6)
        effects = []
        if used.levels_completed != context.levels_completed or key(used) != before_key:
            effects.append(("direct", used.levels_completed))
        for action in (1, 2, 3, 4):
            child = used.clone()
            child.step(action)
            if (child.levels_completed, key(child)) != expected[action]:
                effects.append((action, child.levels_completed))
        print("L3 bare_use", context_name, route, effects)
    for root_name, root in (("raw", env), ("selected", selected)):
        for action in (1, 2, 3, 4):
            node = p.replay(root, [action] * 40)
            print(
                "L3 exhaust",
                root_name,
                action,
                node.levels_completed,
                node.terminal(),
                p.color_counts(node.frame()).get(14, 0),
            )
        for route_name, actions in (
            ("vertical_loop", [1, 2] * 40),
            ("horizontal_loop", [3, 4] * 40),
        ):
            node = p.replay(root, actions)
            print(
                "L3 exhaust_loop",
                root_name,
                route_name,
                node.levels_completed,
                node.terminal(),
                p.color_counts(node.frame()).get(14, 0),
            )
    landmark_routes = {
        "right_then_bottom": [4, 4, 4, 3, 3, 3, 3, 2, 2, 2],
        "bottom_right_bottom": [
            3, 3, 2, 2, 2,
            1, 1, 4, 4, 4, 4, 4,
            3, 3, 3, 3, 1, 1, 2, 2, 2,
        ],
    }
    for name, actions in landmark_routes.items():
        node = p.replay(selected, actions)
        blocks = [
            (b.area, b.bbox)
            for b in p.connected_components(node.frame(), colors=(13,), min_area=1)
        ]
        print(
            "L3 landmarks",
            name,
            node.levels_completed,
            node.terminal(),
            blocks,
        )
    staged = p.replay(env, [3, 3, 2, 2, 2])
    for x, y in points:
        staged.step(6, x, y)
    for action in (2, 2, 2):
        if staged.terminal() or staged.levels_completed > env.levels_completed:
            break
        staged.step(action)
    print(
        "L3 contact_then_grid",
        staged.levels_completed,
        staged.terminal(),
        [
            (b.area, b.bbox)
            for b in p.connected_components(staged.frame(), colors=(13,), min_area=1)
        ],
    )
    timed_path = p.bounded_bfs(
        selected,
        p.level_goal(selected.levels_completed),
        actions=(1, 2, 3, 4),
        max_states=50000,
        max_depth=32,
    )
    print("L3 timed_bfs", timed_path)

    cells = [
        (x, y)
        for y in (50, 55, 60)
        for x in (25, 30, 35)
    ]
    pattern_hits = []
    initial_barrier = p.arr(env.frame())[34:37, 27:31].tobytes()
    for mask in range(1 << len(cells)):
        node = env.clone()
        for bit, (x, y) in enumerate(cells):
            if mask & (1 << bit):
                node.step(6, x, y)
        for action in (3, 3, 2, 2, 2, 2, 3):
            if node.terminal() or node.levels_completed > env.levels_completed:
                break
            node.step(action)
        barrier_changed = (
            p.arr(node.frame())[34:37, 27:31].tobytes() != initial_barrier
        )
        if barrier_changed or node.levels_completed > env.levels_completed:
            pattern_hits.append(
                (
                    mask,
                    node.levels_completed,
                    barrier_changed,
                    [
                        int(node.frame()[y][x])
                        for x, y in cells
                    ],
                )
            )
            if node.levels_completed > env.levels_completed:
                break
    print("L3 pattern_hits", pattern_hits[:12])

    device_hits = []
    for mask in range(1 << len(cells)):
        node = env.clone()
        for bit, (x, y) in enumerate(cells):
            if mask & (1 << bit):
                node.step(6, x, y)
        for action in (4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 3):
            if node.terminal() or node.levels_completed > env.levels_completed:
                break
            node.step(action)
        barrier_changed = (
            p.arr(node.frame())[34:37, 27:31].tobytes() != initial_barrier
        )
        if barrier_changed or node.levels_completed > env.levels_completed:
            device_hits.append((mask, node.levels_completed, barrier_changed))
            if node.levels_completed > env.levels_completed:
                break
    print("L3 device_hits", device_hits[:12])

    remote_click_hits = []
    right_context = p.replay(selected, [4, 4, 4])
    for point_name, point in click_points.items():
        node = right_context.clone()
        node.step(6, *point)
        for action in (3, 3, 3, 3, 2, 2, 2, 2, 3):
            if node.terminal() or node.levels_completed > env.levels_completed:
                break
            node.step(action)
        barrier_changed = (
            p.arr(node.frame())[34:37, 27:31].tobytes() != initial_barrier
        )
        if barrier_changed or node.levels_completed > env.levels_completed:
            remote_click_hits.append(
                (point_name, node.levels_completed, barrier_changed)
            )
    print("L3 remote_click_hits", remote_click_hits)

    control_points = {
        "clue_top": (15, 52),
        "clue_middle": (15, 55),
        "clue_bottom": (15, 58),
        "clue_background": (12, 52),
        "ring": (55, 22),
        "core": (56, 24),
    }
    expected_root = {}
    root_key = key(selected)
    for action in (1, 2, 3, 4):
        child = selected.clone()
        child.step(action)
        expected_root[action] = (child.levels_completed, key(child))
    for name, point in control_points.items():
        node = selected.clone()
        node.step(6, *point)
        effects = []
        if key(node) != root_key or node.levels_completed != selected.levels_completed:
            effects.append(("direct", node.levels_completed))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            if (child.levels_completed, key(child)) != expected_root[action]:
                effects.append((action, child.levels_completed))
        print("L3 control_click", name, point, effects)

    clue_sequence = selected.clone()
    for point in ((15, 52), (15, 55), (15, 58)):
        clue_sequence.step(6, *point)
    for action in (3, 3, 2, 2, 2, 2, 3):
        if (
            clue_sequence.terminal()
            or clue_sequence.levels_completed > env.levels_completed
        ):
            break
        clue_sequence.step(action)
    print(
        "L3 clue_then_exit",
        clue_sequence.levels_completed,
        [
            (b.area, b.bbox)
            for b in p.connected_components(
                clue_sequence.frame(), colors=(13,), min_area=1
            )
        ],
    )


def probe_known_solution(env, movement_action, max_steps, prime=False):
    """Temporary clone probe locating the reward transition on known levels."""
    import perception as p

    node = env.clone()
    starting_level = node.levels_completed

    def report(stage, before, old_level):
        if node.levels_completed <= old_level:
            return False
        objects = [
            (b.color, b.area, b.bbox)
            for b in p.connected_components(before, min_area=4)
            if b.color not in (5, 14)
        ]
        print(
            "KNOWN reward",
            old_level + 1,
            stage,
            "pre_objects",
            objects,
        )
        return True

    if prime:
        before = node.frame()
        old_level = node.levels_completed
        node.step(1)
        report("prime", before, old_level)
    points = [
        (x, y)
        for y in (50, 55, 60)
        for x in (25, 30, 35)
        if int(node.frame()[y][x]) == 0
    ]
    for point in points:
        before = node.frame()
        old_level = node.levels_completed
        node.step(6, *point)
        if report(("click", point), before, old_level):
            return
    for index in range(max_steps):
        before = node.frame()
        old_level = node.levels_completed
        node.step(movement_action)
        if report(("move", movement_action, index + 1), before, old_level):
            return
    print("KNOWN no_reward", starting_level + 1, node.levels_completed)


def probe_known_click_variants(env, movement_action, max_steps, prime=False):
    """Temporary test of which visible grid subset enables a known exit."""
    for label, wanted in (("none", None), ("zeros", 0), ("twos", 2), ("all", "all")):
        node = env.clone()
        if prime:
            node.step(1)
        if wanted is not None:
            frame = node.frame()
            points = [
                (x, y)
                for y in (50, 55, 60)
                for x in (25, 30, 35)
                if wanted == "all" or int(frame[y][x]) == wanted
            ]
            for x, y in points:
                node.step(6, x, y)
        start = node.levels_completed
        for _ in range(max_steps):
            if node.terminal() or node.levels_completed > start:
                break
            node.step(movement_action)
        print("KNOWN click_variant", start + 1, label, node.levels_completed)
