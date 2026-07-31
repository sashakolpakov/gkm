"""Model level-6 autonomous board changes under inert and panel clicks."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import find_right_segment_panel
from perception import arr, frame_delta


def cyan_grid(frame):
    pixels = arr(frame)
    return tuple(
        tuple(
            int((pixels[row : row + 4, col : col + 4] == 11).sum())
            for col in range(33, 61, 4)
        )
        for row in range(4, 32, 4)
    )


def yellow_cells(frame):
    pixels = arr(frame)
    return tuple(
        (int(row), int(col))
        for row, col in zip(*(pixels[:32, :32] == 4).nonzero())
    )


def summary(env):
    frame = arr(env.frame())
    cells = yellow_cells(frame)
    yellow_bbox = (
        min(row for row, _ in cells),
        min(col for _, col in cells),
        max(row for row, _ in cells),
        max(col for _, col in cells),
    )
    return {
        "level": env.levels_completed,
        "clock": {
            color: int((frame[1, 1:62] == color).sum())
            for color in (3, 9)
        },
        "cyan": cyan_grid(frame),
        "yellow": (
            yellow_bbox,
            tuple(
                "".join(
                    "#" if int(frame[row][col]) == 4 else "."
                    for col in range(yellow_bbox[1], yellow_bbox[3] + 1)
                )
                for row in range(yellow_bbox[0], yellow_bbox[2] + 1)
            ),
        ),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    clone = env.clone()
    print("tick", 0, summary(clone))
    for tick in range(1, 13):
        clone.step(6, 63, 63)
        print("tick", tick, summary(clone))

    inert = env.clone()
    panel = env.clone()
    _, rows, cols = find_right_segment_panel(env.frame())
    inert.step(6, 63, 63)
    panel.step(6, cols[0], rows[0])
    print(
        "context",
        {
            "inert": summary(inert),
            "panel": summary(panel),
            "inert_delta": frame_delta(env.frame(), inert.frame()),
            "panel_delta": frame_delta(env.frame(), panel.frame()),
            "between": frame_delta(inert.frame(), panel.frame()),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
