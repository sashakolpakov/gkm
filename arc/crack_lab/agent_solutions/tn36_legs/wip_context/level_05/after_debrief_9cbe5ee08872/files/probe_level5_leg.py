"""Verify the composed level-5 leg on one pristine clone."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    encode_route_resize_rotate_and_acquire_socket,
    find_right_segment_panel,
)
from perception import connected_components


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    clone = env.clone()
    before = clone.levels_completed
    encoded = encode_route_resize_rotate_and_acquire_socket(clone)
    frame = clone.frame()
    _, rows, cols = find_right_segment_panel(frame)
    patterns = tuple(
        tuple(index for index, row in enumerate(rows) if int(frame[row][col]) == 5)
        for col in cols
    )
    reverse_codes = {tuple(value): key for key, value in PROTOCOL_ROWS.items()}
    protocol = "".join(reverse_codes.get(pattern, "?") for pattern in patterns)
    discs = [
        (blob.bbox, blob.area, blob.centroid)
        for blob in connected_components(frame, colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if encoded:
        click_largest_color_9_submit_disc(clone)
    print(
        "candidate",
        {
            "encoded": encoded,
            "protocol": protocol,
            "discs": discs,
            "level_delta": clone.levels_completed - before,
            "parent_level": env.levels_completed,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
