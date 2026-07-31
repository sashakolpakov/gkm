"""Run one level-5 protocol in a fresh foreground arena process."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from perception import arr, color_counts, frame_delta


ROWS = (33, 36, 39, 42, 45, 48)
COLS = (34, 39, 44, 49, 54, 59)
CODES = {
    "N": (),
    "R": (1,),
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "X": (3,),
    "M": (0, 3),
    "A": (0, 2),
    "B": (0, 1, 2, 3, 4, 5),
}
PROGRAM = sys.argv[1] if len(sys.argv) > 1 else "DDDXXB"


def metrics(frame, initial):
    board = arr(frame)[4:32, 33:61]
    base = initial[4:32, 33:61]
    cyan = board == 11
    target = base == 15
    rows, cols = np.where(cyan)
    return {
        "cyan": int(cyan.sum()),
        "bbox": None
        if not len(rows)
        else (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
        "target_covered": int((cyan & target).sum()),
        "cyan_off_target": int((cyan & ~target).sum()),
        "target_remaining": int((board == 15).sum()),
        "board_changed": int((board != base).sum()),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    initial = arr(env.frame()).copy()
    for col, symbol in zip(COLS, PROGRAM):
        for row_index in CODES[symbol]:
            env.step(6, col, ROWS[row_index])
    preview = arr(env.frame()).copy()
    print("preview", PROGRAM, metrics(preview, initial))
    before_level = env.levels_completed
    env.step(6, 57, 58)
    final = arr(env.frame()).copy()
    print(
        "submit",
        {
            "level_delta": env.levels_completed - before_level,
            "metrics": metrics(final, initial),
            "delta": frame_delta(preview, final),
            "colors": color_counts(final),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
