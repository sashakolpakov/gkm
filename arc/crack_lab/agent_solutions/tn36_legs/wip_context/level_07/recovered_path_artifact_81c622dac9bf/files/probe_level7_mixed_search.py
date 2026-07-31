"""Search one transform plus five directions after level-7's first checker."""

import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, connected_components


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
WALLS = {
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 3),
    (3, 0),
    (5, 4),
    (6, 4),
}
INTERESTING = {(1, 2), (1, 5), (2, 0), (3, 1), (3, 6), (5, 5), (6, 2)}


def panel_geometry(frame):
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] > len(frame[0]) / 2
    ]
    return (
        sorted({int(round(blob.centroid[0])) for blob in segments}),
        sorted({int(round(blob.centroid[1])) for blob in segments}),
    )


def set_program(env, rows, cols, program, codes):
    frame = arr(env.frame()).copy()
    for col, symbol in zip(cols, program):
        for row_index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (row_index in codes[symbol]):
                env.step(6, col, row)


def simulate(program, model, start):
    position = start
    for symbol in program:
        dr, dc = DIRS[symbol]
        entered_from = position
        neighbor = (position[0] + dr, position[1] + dc)
        if not (0 <= neighbor[0] < 7 and 0 <= neighbor[1] < 7):
            return None
        if neighbor in WALLS:
            return None
        position = neighbor
        if model == "swap_cell":
            if position == (3, 6):
                position = (3, 1)
            elif position == (3, 1):
                position = (3, 6)
        elif model == "exit_inside":
            if position == (3, 6):
                position = (3, 2)
            elif position == (3, 1):
                position = (3, 5)
        elif model == "horizontal":
            if position == (3, 6) and entered_from == (3, 5):
                position = (3, 2)
            elif position == (3, 1) and entered_from == (3, 2):
                position = (3, 5)
    return position


def large_cyan(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(11,), min_area=9)
        if blob.bbox[2] < 32 and blob.bbox[1] >= 32
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    codes = dict(PROTOCOL_ROWS)
    codes.update(learn_direction_protocol_from_selector(env))
    rows, cols = panel_geometry(env.frame())
    set_program(env, rows, cols, "UURRRD", codes)
    click_largest_color_9_submit_disc(env)
    for selector_col in (5, 15, 25, 35):
        env.step(6, selector_col, 58)
    root_frame = arr(env.frame()).copy()
    root_cyan = large_cyan(root_frame)

    transform_count = int(os.environ.get("GKM_TRANSFORM_COUNT", "1"))
    direction_count = 6 - transform_count
    routes = set()
    for symbols in itertools.product(DIRS, repeat=direction_count):
        route = "".join(symbols)
        if any(
            simulate(route, model, start) in INTERESTING
            for model in ("normal", "swap_cell", "exit_inside", "horizontal")
            for start in ((5, 5), (1, 2))
        ):
            routes.add(route)

    programs = set()
    for transforms in itertools.product(
        os.environ.get("GKM_TRANSFORMS", "MX"),
        repeat=transform_count,
    ):
        for route in routes:
            for positions in itertools.combinations(range(6), transform_count):
                route_index = 0
                transform_index = 0
                program = []
                for index in range(6):
                    if index in positions:
                        program.append(transforms[transform_index])
                        transform_index += 1
                    else:
                        program.append(route[route_index])
                        route_index += 1
                programs.add("".join(program))
    print(
        "candidate_count",
        {"routes": len(routes), "programs": len(programs)},
        flush=True,
    )

    hits = []
    for program in sorted(programs):
        clone = env.clone()
        set_program(clone, rows, cols, program, codes)
        click_largest_color_9_submit_disc(clone)
        final = arr(clone.frame())
        cyan = large_cyan(final)
        if cyan != root_cyan or clone.levels_completed > env.levels_completed:
            hits.append(
                {
                    "program": program,
                    "large_cyan": cyan,
                    "level": clone.levels_completed,
                }
            )
    print(
        "mixed_search",
        {
            "root_unchanged": bool((root_frame == arr(env.frame())).all()),
            "hits": hits,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
