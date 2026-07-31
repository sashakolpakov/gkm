import sys
import itertools
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
import players


def reach_level_4(env):
    for k in (1, 2, 3):
        getattr(players, f"play_level_{k}")(env)


def probe(env):
    reach_level_4(env)
    symbols = {0: ".", 2: "#", 3: "W", 5: " ", 6: "g", 9: "a",
               10: "A", 13: "b", 14: "x", 15: "*"}
    frame = P.arr(env.frame())
    print("MAP")
    for r in range(0, 64, 2):
        row = []
        for c in range(0, 64, 2):
            vals, counts = __import__("numpy").unique(
                frame[r:r + 2, c:c + 2], return_counts=True
            )
            ranked = sorted(
                zip(counts, vals), key=lambda item: (item[1] != 5, item[0]),
                reverse=True
            )
            row.append(symbols.get(int(ranked[0][1]), "?"))
        print("".join(row))
    for label, (r0, c0, r1, c1) in {
        "REFERENCE": (0, 0, 10, 10),
        "EXAMPLE": (11, 0, 21, 10),
        "PANEL_RAW": (47, 22, 64, 39),
    }.items():
        print(label)
        for row in frame[r0:r1, c0:c1]:
            print("".join(symbols.get(int(v), "?") for v in row))
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COLORS", P.color_counts(env.frame()))
    print("OBJECTS")
    for obj in P.object_candidates(env.frame(), min_area=3):
        print(obj["color"], obj["bbox"], obj["area"])
    print("DELTAS")
    for action, delta in P.action_deltas(env, actions=env.actions).items():
        print(action, delta["count"], delta["bbox"], delta["samples"][:12])
    xs = (25, 30, 35)
    ys = (50, 55, 60)

    def panel(node):
        frame = node.frame()
        return tuple(int(frame[y][x]) for y in ys for x in xs)

    print("PANEL", panel(env))
    for y in ys:
        for x in xs:
            node = env.clone()
            node.step(6, x, y)
            print("CLICK", x, y, panel(node), node.levels_completed)

    path = None
    print("PANEL_PATH", path)
    if path:
        node = env.clone()
        for action in path:
            node.step(*action)
        print("AFTER_PANEL", node.levels_completed, panel(node), P.color_counts(node.frame()))
        print("AFTER_OBJECTS")
        for obj in P.object_candidates(node.frame(), min_area=3):
            print(obj["color"], obj["bbox"], obj["area"])
    for actions in ([2, 2, 4, 4], [4, 4, 2, 2]):
        node = P.replay(env, actions)
        print(
            "MOVE_TEST", actions, node.levels_completed, panel(node),
            P.frame_delta(env.frame(), node.frame())["bbox"],
            P.color_counts(node.frame()),
        )
    move_path = None
    print("MOVE_WIN", move_path)
    routes = {
        "target_then_right": [2, 2, 4, 4, 2, 2, 4, 4],
        "target_then_left": [2, 2, 4, 4] + [3] * 10,
        "left_direct": [2, 2] + [3] * 10,
    }
    for name, actions in routes.items():
        node = P.replay(env, actions)
        interesting = [
            (b.color, b.bbox, b.area)
            for b in P.connected_components(
                node.frame(), colors=(6, 9, 10, 13, 14, 15), min_area=3
            )
        ]
        print("ROUTE", name, node.levels_completed, panel(node), interesting)
    for name, actions in {
        "left_trace": [2, 2, 4, 4] + [3] * 10,
        "right_trace": [2, 2, 4, 4, 2, 2] + [4] * 4,
    }.items():
        node = env.clone()
        print("TRACE", name)
        for i, action in enumerate(actions, 1):
            before = node.frame()
            node.step(action)
            movers = [
                b.bbox for b in P.connected_components(
                    node.frame(), colors=(9, 10), min_area=4
                )
                if b.bbox[1] < 47
            ]
            delta = P.frame_delta(before, node.frame())
            print(i, action, movers, delta["count"], delta["bbox"])
    def movement_key(node):
        blobs = P.connected_components(
            node.frame(), colors=(6, 9, 10, 13, 14), min_area=3
        )
        avatar = tuple(
            b.bbox for b in blobs
            if b.color in (9, 10) and b.bbox[1] < 47 and b.bbox[0] < 34
        )
        marker = any(b.color == 14 and b.bbox == (28, 44, 29, 45)
                     for b in blobs)
        return avatar, marker

    pure_path = P.bounded_replay_bfs(
        env,
        goal_fn=P.level_goal(env.levels_completed),
        action_fn=lambda _: (1, 2, 3, 4),
        key_fn=movement_key,
        max_states=500,
        max_depth=24,
    )
    print("SYMBOLIC_MOVE_WIN", pure_path)
    click_all = [(6, x, y) for y in ys for x in xs]
    for name, moves in {
        "click_all": [],
        "target_click_all": [2, 2, 4, 4],
        "click_all_target": None,
    }.items():
        node = env.clone()
        sequence = click_all + [2, 2, 4, 4] if moves is None else moves + click_all
        for action in sequence:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
        print("COMPOSE", name, node.levels_completed, panel(node),
              P.color_counts(node.frame()))
    plus = [(6, 30, 50), (6, 25, 55), (6, 35, 55), (6, 30, 60)]
    for name, sequence in {
        "plus": plus,
        "target_plus": [2, 2, 4, 4] + plus,
        "prime_select": [6] + plus,
        "target_prime_select": [2, 2, 4, 4, 6] + plus,
    }.items():
        node = env.clone()
        for action in sequence:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
        print("PLUS", name, node.levels_completed, panel(node),
              P.color_counts(node.frame()))
    root = P.replay(env, [2, 2, 4, 4])
    outcomes = Counter()
    wins = []
    coords = [(x, y) for y in ys for x in xs]
    for combo in itertools.combinations(coords, 4):
        node = root.clone()
        for x, y in combo:
            node.step(6, x, y)
        pieces = P.connected_components(
            node.frame(), colors=(9, 10), min_area=1
        )
        remote = tuple(
            (b.color, b.bbox, b.area)
            for b in pieces
            if b.bbox[1] >= 47
        )
        main = tuple(
            (b.color, b.bbox, b.area) for b in pieces if b.bbox[1] < 47
        )
        outcomes[(main, remote)] += 1
        if node.levels_completed > env.levels_completed:
            wins.append(combo)
    print("COMMAND_OUTCOMES", len(outcomes), outcomes.most_common(8))
    print("COMMAND_WINS", wins)
    shaped = env.clone()
    shaped_sequence = [2, 2, 4, 4] + plus
    for action in shaped_sequence:
        shaped.step(*action) if isinstance(action, tuple) else shaped.step(action)

    def world_key(node):
        return tuple(
            (b.color, b.bbox, b.area)
            for b in P.connected_components(
                node.frame(), colors=(6, 9, 10, 13, 15), min_area=1
            )
        )

    shaped_win = None
    print("SHAPED_MOVE_WIN", shaped_win)
    def fixed_parts(node):
        blobs = P.connected_components(
            node.frame(), colors=(6, 9, 10, 13), min_area=1
        )
        return tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.color in (6, 13) or b.bbox[1] >= 47
        )

    base_fixed = fixed_parts(shaped)
    interaction_path = None
    print("INTERACTION_PATH", interaction_path)
    if interaction_path:
        node = P.replay(shaped, interaction_path)
        print("INTERACTION_PARTS", fixed_parts(node))
    for name, actions in {
        "small_left": [3] * 12,
        "small_down_right": [2] * 3 + [4] * 5,
        "small_up": [1] * 8,
    }.items():
        node = shaped.clone()
        trace = []
        for action in actions:
            node.step(action)
            main = tuple(
                b.bbox for b in P.connected_components(
                    node.frame(), colors=(9, 10), min_area=1
                )
                if b.bbox[1] < 47
            )
            trace.append(main)
        print("SMALL_TRACE", name, trace)
    def main_bounds(node):
        parts = [
            b.bbox for b in P.connected_components(
                node.frame(), colors=(9, 10), min_area=1
            )
            if b.bbox[1] < 47
        ]
        return (
            min(b[0] for b in parts), min(b[1] for b in parts),
            max(b[2] for b in parts), max(b[3] for b in parts),
        )

    left_path = P.bounded_replay_bfs(
        shaped,
        goal_fn=lambda node, _: main_bounds(node)[1] <= 12,
        action_fn=lambda _: (1, 2, 3, 4),
        key_fn=main_bounds,
        max_states=500,
        max_depth=22,
    )
    print("LEFT_REACH", left_path)
    if left_path:
        node = P.replay(shaped, left_path)
        for action in (3, 3, 1, 2, 4):
            before = main_bounds(node)
            node.step(action)
            print("AT_LEFT", action, before, main_bounds(node),
                  node.levels_completed, fixed_parts(node))
        for name, command in {
            "vertical": [(6, 30, 50), (6, 30, 55), (6, 30, 60)],
            "horizontal": [(6, 25, 55), (6, 30, 55), (6, 35, 55)],
        }.items():
            test = P.replay(shaped, left_path)
            for action in command:
                test.step(*action)
            print("PORTAL_COMMAND", name, test.levels_completed,
                  main_bounds(test), panel(test), fixed_parts(test),
                  P.color_counts(test.frame()))
        opened = P.replay(shaped, left_path)
        for action in [(6, 30, 50), (6, 30, 55), (6, 30, 60)]:
            opened.step(*action)
        for i in range(1, 6):
            opened.step(3)
            agents = tuple(
                (b.color, b.bbox, b.area)
                for b in P.connected_components(
                    opened.frame(), colors=(9, 10), min_area=1
                )
            )
            print("AFTER_OPEN_LEFT", i, opened.levels_completed, agents)
        for label, base in {
            "initial": env,
            "opened": P.replay(shaped, left_path),
        }.items():
            test = base.clone()
            if label == "opened":
                for action in [(6, 30, 50), (6, 30, 55), (6, 30, 60)]:
                    test.step(*action)
            before = test.frame()
            test.step(6, 53, 37)
            click_delta = P.frame_delta(before, test.frame())
            before_move = test.frame()
            test.step(3)
            move_delta = P.frame_delta(before_move, test.frame())
            agents = tuple(
                (b.color, b.bbox, b.area)
                for b in P.connected_components(
                    test.frame(), colors=(9, 10), min_area=1
                )
            )
            print("SELECT_REMOTE", label, click_delta["count"],
                  move_delta["count"], agents)
        travel = P.replay(shaped, left_path)
        for action in [(6, 30, 50), (6, 30, 55), (6, 30, 60)]:
            travel.step(*action)
        travel.step(3)
        travel.step(3)
        finish_path = None
        print("FINISH_PATH", finish_path)
        right_reach = P.bounded_replay_bfs(
            travel,
            goal_fn=lambda node, _: main_bounds(node)[1] >= 49,
            action_fn=lambda _: (1, 2, 3, 4),
            key_fn=main_bounds,
            max_states=1000,
            max_depth=50,
        )
        print("RIGHT_REACH", right_reach)


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)
