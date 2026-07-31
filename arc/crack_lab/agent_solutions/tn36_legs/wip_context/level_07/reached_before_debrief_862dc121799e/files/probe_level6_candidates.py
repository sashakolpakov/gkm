"""Verify existing configuration legs against pristine level-6 entry."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    encode_reacquisition_route_through_barrier,
    encode_route_resize_rotate_and_acquire_socket,
    find_right_segment_panel,
    reacquire_max_scaled_agent_and_route_to_socket,
)
from perception import arr, color_counts, frame_delta


def panel_symbols(frame):
    _, rows, cols = find_right_segment_panel(frame)
    inverse = {tuple(value): key for key, value in PROTOCOL_ROWS.items()}
    return "".join(
        inverse.get(
            tuple(index for index, row in enumerate(rows) if int(frame[row][col]) == 5),
            "?",
        )
        for col in cols
    )


def board_grid(frame):
    lines = []
    for row in range(4, 32, 4):
        cells = []
        for col in range(33, 61, 4):
            counts = Counter(int(value) for value in arr(frame)[row : row + 4, col : col + 4].flat)
            cells.append("/".join(f"{color}:{count}" for color, count in sorted(counts.items())))
        lines.append(cells)
    return lines


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    print("board_grid", board_grid(env.frame()))
    base_level = env.levels_completed
    candidates = (
        ("barrier_route", encode_reacquisition_route_through_barrier),
        ("max_reacquire", reacquire_max_scaled_agent_and_route_to_socket),
        ("shape_socket", encode_route_resize_rotate_and_acquire_socket),
    )
    for name, configure in candidates:
        clone = env.clone()
        before = arr(clone.frame()).copy()
        configured = configure(clone)
        after_config = arr(clone.frame()).copy()
        symbols = panel_symbols(after_config)
        click_largest_color_9_submit_disc(clone)
        after_submit = arr(clone.frame()).copy()
        print(
            "candidate",
            {
                "name": name,
                "configured": configured,
                "symbols": symbols,
                "config_delta": frame_delta(before, after_config)["count"],
                "submit_delta": frame_delta(after_config, after_submit),
                "level_delta": clone.levels_completed - base_level,
                "colors": color_counts(after_submit),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
