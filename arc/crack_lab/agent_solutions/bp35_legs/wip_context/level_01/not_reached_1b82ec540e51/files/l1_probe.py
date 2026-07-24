import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import color_counts, connected_components, frame_delta


ACTIONS = (3, 4, 6, 7)


def compact_blobs(frame, min_area=4):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=min_area)
    ]


def color_runs(frame, colors=(3, 9, 10, 11, 14)):
    rows = []
    for r, row in enumerate(frame):
        runs = []
        c = 0
        while c < len(row):
            color = int(row[c])
            end = c
            while end + 1 < len(row) and int(row[end + 1]) == color:
                end += 1
            if color in colors:
                runs.append((color, c, end))
            c = end + 1
        rows.append(tuple(runs))
    groups = []
    start = 0
    for r in range(1, len(rows) + 1):
        if r == len(rows) or rows[r] != rows[start]:
            if rows[start]:
                groups.append((start, r - 1, rows[start]))
            start = r
    return groups


def actors(env):
    blobs = connected_components(env.frame(), colors=(9, 11, 14), min_area=1)
    return [(b.color, b.bbox, b.area) for b in blobs]


def trace(env, name, actions):
    clone = env.clone()
    print("trace", name, "start", actors(clone))
    for i, action in enumerate(actions, 1):
        clone.step(action)
        print(
            " t",
            i,
            action,
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "actors",
            actors(clone),
        )
        if clone.terminal() or clone.levels_completed:
            break


def click_probe(env, name, x, y):
    clone = env.clone()
    before = clone.frame().copy()
    try:
        clone.step(6, x, y)
        delta = frame_delta(before, clone.frame())
        print(
            "click",
            name,
            (x, y),
            "level",
            clone.levels_completed,
            "delta",
            (delta["count"], delta["bbox"]),
            "colors",
            color_counts(clone.frame()),
            "actors",
            actors(clone),
        )
    except Exception as exc:
        print("click", name, (x, y), "error", type(exc).__name__, str(exc))


def click_trace(env, name, points):
    clone = env.clone()
    print("click_trace", name)
    for i, (x, y) in enumerate(points, 1):
        clone.step(6, x, y)
        yellow = connected_components(clone.frame(), colors=(14,), min_area=1)
        print(
            " c",
            i,
            (x, y),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "yellow",
            [(b.bbox, b.area) for b in yellow],
        )
        if clone.terminal() or clone.levels_completed:
            break


def route_trace(env, name, actions):
    clone = env.clone()
    print("route", name)
    for i, action in enumerate(actions, 1):
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
        body = connected_components(clone.frame(), colors=(9,), min_area=1)
        yellow = connected_components(clone.frame(), colors=(14,), min_area=1)
        print(
            " r",
            i,
            action,
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "body",
            [b.bbox for b in body],
            "yellow_n",
            len(yellow),
        )
        if clone.terminal() or clone.levels_completed:
            break


def snapshot(env, name, actions):
    clone = env.clone()
    for action in actions:
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
    print("snapshot", name, "level", clone.levels_completed, "colors", color_counts(clone.frame()))
    print(" smap", color_runs(clone.frame()))


def clear_current_yellow(clone):
    points = [
        (round(b.centroid[1]), round(b.centroid[0]))
        for b in connected_components(clone.frame(), colors=(14,), min_area=4)
    ]
    for x, y in points:
        clone.step(6, x, y)
    return points


def phase_probe(env, name, pre_actions, rounds=1):
    clone = env.clone()
    path = []
    for action in pre_actions:
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
        path.append(action)
    for round_no in range(1, rounds + 1):
        points = clear_current_yellow(clone)
        path.extend((6, x, y) for x, y in points)
        remaining = connected_components(clone.frame(), colors=(14,), min_area=4)
        print(
            "phase",
            name,
            round_no,
            "clicked",
            points,
            "remaining",
            len(remaining),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "moves",
            len(path),
        )


def probe(env):
    print("actions", tuple(env.actions))
    print("colors", color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    print("map", color_runs(env.frame()))
    for action in ACTIONS:
        clone = env.clone()
        before = compact_blobs(clone.frame())
        frame_before = clone.frame().copy()
        try:
            clone.step(action)
            delta = frame_delta(frame_before, clone.frame())
            after = compact_blobs(clone.frame())
            print(
                "action",
                action,
                "reward",
                clone.levels_completed,
                "delta",
                (delta["count"], delta["bbox"]),
                "blob_changes",
                sorted(set(before) ^ set(after)),
            )
        except Exception as exc:
            print("action", action, "error", type(exc).__name__, str(exc))
    trace(env, "left12", [3] * 12)
    trace(env, "right12", [4] * 12)
    trace(env, "use12", [6] * 12)
    trace(env, "left_use", [3] * 4 + [6] * 4 + [3] * 4)
    trace(env, "right_use", [4] * 4 + [6] * 4 + [4] * 4)
    for name, x, y in (
        ("yellow_top", 15, 3),
        ("yellow_mid", 33, 15),
        ("avatar", 21, 39),
        ("walkable", 40, 38),
        ("other_region", 8, 8),
        ("bottom_left", 0, 63),
    ):
        click_probe(env, name, x, y)
    click_trace(
        env,
        "all_initial_yellow",
        [(15, 3), (21, 3), (27, 3), (33, 15), (39, 15), (45, 15), (51, 15)],
    )
    all_yellow = [
        (6, 15, 3),
        (6, 21, 3),
        (6, 27, 3),
        (6, 33, 15),
        (6, 39, 15),
        (6, 45, 15),
        (6, 51, 15),
    ]
    route_trace(env, "clear_then_right", all_yellow + [4] * 12)
    for point in all_yellow:
        route_trace(env, f"click_{point[1]}_{point[2]}_then_right", [point] + [4] * 7)
    route_trace(env, "right4_left6", [4] * 4 + [3] * 6)
    snapshot(env, "right4", [4] * 4)
    snapshot(env, "right4_left4", [4] * 4 + [3] * 4)
    phase_probe(env, "right4_clear", [4] * 4)
    phase_probe(env, "clear_right4_clear", all_yellow + [4] * 4)


print("run", A.run_program("bp35", probe))
