import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(frame, min_area=4)
        if o["color"] != 5
    ]


def symbolic_map(frame):
    chars = {5: ".", 6: "L", 7: "R", 8: "#", 9: "g", 10: "A", 12: "c", 14: "d"}
    sigs = P.block_signatures(frame, cell=4)
    rows = []
    for r in range(16):
        row = []
        for c in range(16):
            sig = sigs[(r, c)]
            non_bg = [v for v in sig if v != 5]
            row.append(chars.get(non_bg[-1], "?") if non_bg else ".")
        rows.append("".join(row))
    return rows


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COUNTS", P.color_counts(env.frame()))
    print("OBJECTS", compact_objects(env.frame()))
    print("MAP")
    for row in symbolic_map(env.frame()):
        print(row)
    base = P.arr(env.frame()).copy()
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        delta = P.frame_delta(base, clone.frame())
        print(
            "ACTION",
            action,
            "DELTA",
            (delta["count"], delta["bbox"]),
            "LEVELS",
            clone.levels_completed,
            "OBJECTS",
            compact_objects(clone.frame()),
        )
    path = P.bounded_bfs(
        env,
        P.level_goal(5),
        actions=(1, 2, 3, 4, 5),
        max_states=5000,
        max_depth=80,
    )
    print("REWARD_PATH", path)
    for obj in compact_objects(env.frame()):
        color, (r0, c0, r1, c1), area = obj
        if color not in (9, 10, 12, 14):
            continue
        x, y = (c0 + c1) // 2, (r0 + r1) // 2
        selected = env.clone()
        before = P.arr(selected.frame()).copy()
        selected.step(6, x, y)
        click_delta = P.frame_delta(before, selected.frame())
        results = []
        for action in (1, 2, 3, 4, 5):
            moved = selected.clone()
            pre_move = P.arr(moved.frame()).copy()
            moved.step(action)
            delta = P.frame_delta(pre_move, moved.frame())
            results.append((action, delta["count"], delta["bbox"]))
        print("SELECT", (color, (x, y), area), "CLICK", click_delta["count"], "MOVES", results)
        if color == 9:
            print("SELECTED_OBJECTS", compact_objects(selected.frame()))
            for action in (1, 2, 3, 4, 5):
                moved = selected.clone()
                moved.step(action)
                print("SELECTED_ACTION", action, compact_objects(moved.frame()))
            for route in ((3, 3, 3, 3, 3), (4, 4, 4, 4, 4)):
                moved = selected.clone()
                for step, action in enumerate(route, 1):
                    moved.step(action)
                    print(
                        "TINY_ROUTE",
                        route[0],
                        step,
                        "LEVELS",
                        moved.levels_completed,
                        "OBJECTS",
                        compact_objects(moved.frame()),
                    )
    for park_steps in range(1, 6):
        staged = env.clone()
        staged.step(6, 31, 43)
        for _ in range(park_steps):
            staged.step(4)
        staged.step(6, 19, 23)
        if park_steps == 1:
            print("RESTAGED_OBJECTS", compact_objects(staged.frame()))
        path = P.bounded_bfs(
            staged,
            P.level_goal(5),
            actions=(1, 2, 3, 4, 5),
            max_states=10000,
            max_depth=80,
        )
        print("STAGED_SEARCH", park_steps, path)


A.run_program("m0r0", probe)
