"""Fresh-replay and reproduce the validated level-4 reacquisition mechanic."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import players
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    find_right_segment_panel,
    reacquire_max_scaled_agent_and_route_to_socket,
)
from perception import arr, connected_components, frame_delta


def mask(frame, bbox, color):
    r0, c0, r1, c1 = bbox
    return tuple(
        "".join("#" if int(frame[row][col]) == color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


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


def cyan_components(frame):
    return tuple(
        (blob.bbox, blob.area, mask(frame, blob.bbox, 11))
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def observe(env):
    for level in (1, 2, 3):
        getattr(players, f"play_level_{level}")(env)
    entry = arr(env.frame()).copy()
    configured = reacquire_max_scaled_agent_and_route_to_socket(env)
    after_config = arr(env.frame()).copy()
    click_largest_color_9_submit_disc(env)
    after_submit = arr(env.frame()).copy()
    print(
        "level4_repro",
        {
            "entry_level": 4,
            "entry_cyan": cyan_components(entry),
            "configured": configured,
            "program": panel_symbols(after_config),
            "config_delta": frame_delta(entry, after_config),
            "submit_delta": frame_delta(after_config, after_submit),
            "levels_completed": env.levels_completed,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
