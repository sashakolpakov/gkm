"""Compact clean-room observations at pristine level-3 entry."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, color_counts, frame_delta


SYMBOL = {0: ".", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 9: "9", 11: "B"}


def transition_counts(before, after):
    changed = arr(before) != arr(after)
    return dict(
        sorted(
            Counter(
                zip(
                    (int(value) for value in arr(before)[changed]),
                    (int(value) for value in arr(after)[changed]),
                )
            ).items()
        )
    )


def show_region(frame, r0, c0, r1, c1):
    region = arr(frame)[r0:r1, c0:c1]
    return ["" .join(SYMBOL[int(value)] for value in row) for row in region]


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    print("entry", {"level": env.levels_completed + 1, "actions": env.actions})
    print("colors", color_counts(env.frame()))
    frame = arr(env.frame())

    print("top_right_4x4_majorities")
    for row in range(0, 32, 4):
        cells = []
        for col in range(32, 64, 4):
            counts = Counter(int(value) for value in frame[row : row + 4, col : col + 4].flat)
            color, count = counts.most_common(1)[0]
            cells.append(f"{color}:{count}")
        print(row, " ".join(cells))

    segment_rows = (33, 36, 39, 42, 45, 48)
    for name, columns in (("left_segments", (8, 13, 18, 23)), ("right_segments", (34, 39, 44, 49, 54, 59))):
        print(name, ["".join(str(int(frame[row, col])) for col in columns) for row in segment_rows])

    for name, c0, c1 in (
        ("bottom_left", 1, 10),
        ("control_1", 11, 20),
        ("control_2", 21, 30),
        ("control_3", 31, 40),
        ("bottom_right", 53, 62),
    ):
        print(name)
        print("\n".join(show_region(frame, 54, c0, 63, c1)))

    tests = (
        ("left_on_segment", 8, 33),
        ("left_off_segment", 8, 39),
        ("right_off_segment", 34, 33),
        ("top_cell", 34, 5),
        ("bottom_left", 5, 58),
        ("control_1", 15, 58),
        ("control_2", 25, 58),
        ("control_3", 35, 58),
        ("bottom_right", 57, 58),
        ("top_banner", 31, 1),
        ("background", 31, 32),
    )
    for name, x, y in tests:
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(6, x, y)
        after = arr(clone.frame()).copy()
        delta = frame_delta(before, after)
        print(
            "click",
            name,
            (x, y),
            {
                "levels": clone.levels_completed,
                "count": delta["count"],
                "bbox": delta["bbox"],
                "transitions": transition_counts(before, after),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
