import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from legs import select_grid_cells_of_color
from players import play_level_1, play_level_2


def compact_blobs(frame):
    return [
        (b.color, b.area, b.bbox, tuple(round(v, 1) for v in b.centroid))
        for b in P.connected_components(frame, min_area=4)
    ]


def modal_cells(frame):
    a = P.arr(frame)
    rows = []
    for y in range(0, 64, 4):
        row = []
        for x in range(0, 64, 4):
            vals, counts = __import__("numpy").unique(a[y:y + 4, x:x + 4], return_counts=True)
            row.append(f"{int(vals[counts.argmax()]):X}")
        rows.append("".join(row))
    return rows


def region(frame, x0, y0, x1, y1):
    a = P.arr(frame)
    return "\n".join(
        "".join(f"{int(a[y, x]):X}" for x in range(x0, x1 + 1))
        for y in range(y0, y1 + 1)
    )


def probe(env):
    play_level_1(env)
    play_level_2(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", P.color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    print("cells")
    print("\n".join(modal_cells(env.frame())))
    print("upper_panel\n" + region(env.frame(), 47, 10, 63, 39))
    print("lower_panel\n" + region(env.frame(), 18, 44, 43, 60))
    print("gate\n" + region(env.frame(), 28, 20, 44, 34))
    effects = {}
    for y in range(44, 61, 2):
        for x in range(18, 44, 2):
            c0 = env.clone()
            before0 = c0.frame()
            c0.step(6, x, y)
            d0 = P.frame_delta(before0, c0.frame())
            if d0["count"]:
                effects[(x, y)] = (d0["count"], d0["bbox"])
    print("click_zones", effects)
    for point in (
        (24, 37), (40, 25), (35, 29), (24, 49), (24, 57),
        (55, 16), (50, 25), (55, 25), (60, 25),
        (50, 30), (55, 30), (60, 30),
        (50, 35), (55, 35), (60, 35),
    ):
        c = env.clone()
        before = c.frame()
        c.step(6, *point)
        delta = P.frame_delta(before, c.frame())
        if delta["count"] or c.levels_completed != env.levels_completed:
            print("click", point, delta, "level", c.levels_completed)
    c = env.clone()
    before = c.frame()
    selected = [
        (x, y)
        for y in (50, 55, 60)
        for x in (25, 30, 35)
        if int(before[y][x]) == 0
    ]
    select_grid_cells_of_color(
        c, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    print(
        "selected_zeros", selected,
        P.frame_delta(before, c.frame()), "level", c.levels_completed,
    )
    routes = {
        "push_once": [2, 2, 3, 3, 2],
        "push_twice": [2, 2, 3, 3, 2, 2],
        "to_upper_right_a": [2, 2, 2, 2, 3, 3, 3],
        "to_upper_right_b": [3, 3, 3, 2, 2, 2, 2],
        "to_bottom": [4, 4, 4, 4, 4],
        "to_obstacle": [2, 2, 2, 3, 3],
        "left_edge": [1] * 8,
        "upper_edge": [3] * 8,
        "right_edge": [2] * 8,
        "lower_edge": [4] * 8,
    }
    for name, actions in routes.items():
        r = P.replay(env, actions)
        print(
            "route", name, "level", r.levels_completed,
            "delta", P.frame_delta(env.frame(), r.frame()),
            "colors", P.color_counts(r.frame()),
        )
    for name, root in (("raw", env), ("selected", c)):
        for actions in ([2, 2, 3, 3, 2], [2, 2, 3, 3, 2, 2]):
            r = P.replay(root, actions)
            print(
                "push", name, actions, "level", r.levels_completed,
                "delta", P.frame_delta(root.frame(), r.frame()),
                "objects", compact_blobs(r.frame()),
            )
    path = P.bounded_bfs(
        env,
        P.level_goal(env.levels_completed),
        actions=(1, 2, 3, 4),
        key_fn=lambda e: P.arr(e.frame())[:62].tobytes(),
        max_states=5000,
        max_depth=32,
    )
    print("movement_bfs", path)
    path2 = P.bounded_bfs(
        c,
        P.level_goal(c.levels_completed),
        actions=(1, 2, 3, 4),
        key_fn=lambda e: P.arr(e.frame())[:62].tobytes(),
        max_states=5000,
        max_depth=32,
    )
    print("post_click_movement_bfs", path2)
    path3 = P.bounded_replay_bfs(
        c,
        P.level_goal(c.levels_completed),
        action_fn=lambda node: (1, 2, 3, 4),
        key_fn=lambda e: P.arr(e.frame())[:62].tobytes(),
        max_states=3000,
        max_depth=32,
    )
    print("post_click_replay_bfs", path3)


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
