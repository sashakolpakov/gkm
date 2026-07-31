"""Fresh-replay and reproduce the validated level-5 transform/acquire program."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import players
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    encode_route_resize_rotate_and_acquire_socket,
    find_right_segment_panel,
)
from perception import arr, connected_components, frame_delta


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


def board_components(frame, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(color,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def observe(env):
    for level in (1, 2, 3, 4):
        getattr(players, f"play_level_{level}")(env)
    entry = arr(env.frame()).copy()
    configured = encode_route_resize_rotate_and_acquire_socket(env)
    after_config = arr(env.frame()).copy()
    click_largest_color_9_submit_disc(env)
    after_submit = arr(env.frame()).copy()
    print(
        "level5_repro",
        {
            "entry_cyan": board_components(entry, 11),
            "entry_white": board_components(entry, 15),
            "configured": configured,
            "program": panel_symbols(after_config),
            "submit_delta": frame_delta(after_config, after_submit)["count"],
            "levels_completed": env.levels_completed,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
