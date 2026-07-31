"""Run one level-7 protocol from a fresh disposable arena."""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, color_counts, connected_components, frame_delta


DIRECTION_ROWS = {
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
}


def panel_geometry(frame):
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] > len(frame[0]) / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    return rows, cols


def transitions(before, after):
    changed = arr(before) != arr(after)
    return tuple(
        sorted(
            Counter(
                zip(
                    (int(v) for v in arr(before)[changed]),
                    (int(v) for v in arr(after)[changed]),
                )
            ).items()
        )
    )


def board_objects(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=range(7, 16), min_area=2)
        if blob.bbox[2] < 32 and blob.bbox[1] >= 32
    )


def cyan_cells(frame):
    pixels = arr(frame)
    return tuple(
        (row_index, col_index, int((pixels[row : row + 4, col : col + 4] == 11).sum()))
        for row_index, row in enumerate(range(4, 32, 4))
        for col_index, col in enumerate(range(33, 61, 4))
        if int((pixels[row : row + 4, col : col + 4] == 11).sum())
    )


def force_relearn_direction_protocol(env):
    frame = env.frame()
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] < len(frame[0]) / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    protocol = {}
    for direction, selector_col in (
        ("D", 5),
        ("U", 15),
        ("R", 25),
        ("L", 35),
    ):
        env.step(6, selector_col, 58)
        current = env.frame()
        patterns = {
            tuple(
                index
                for index, row in enumerate(rows)
                if int(current[row][col]) == 5
            )
            for col in cols
        }
        if len(patterns) != 1:
            return None
        protocol[direction] = patterns.pop()
    return protocol


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    programs = os.environ.get(
        "GKM_PROGRAMS",
        os.environ.get("GKM_PROGRAM", "MUUUUU"),
    ).split(",")
    learned = learn_direction_protocol_from_selector(env)
    before = arr(env.frame()).copy()
    rows, cols = panel_geometry(before)
    codes = dict(PROTOCOL_ROWS)
    codes.update(learned or DIRECTION_ROWS)
    codes["-"] = ()
    print("learned", learned)
    for stage_index, program in enumerate(programs, start=1):
        if stage_index > 1 and os.environ.get("GKM_RELEARN_STAGES") == "1":
            relearned = learn_direction_protocol_from_selector(env)
            if relearned is None:
                relearned = force_relearn_direction_protocol(env)
            codes.update(relearned or {})
            print("relearned", {"stage": stage_index, "codes": relearned})
        selection = None
        selection_text = os.environ.get(f"GKM_SELECT_STAGE{stage_index}")
        if selection_text:
            selection = tuple(int(value) for value in selection_text.split(":"))
            env.step(6, *selection)
        if len(program) != len(cols) or any(symbol not in codes for symbol in program):
            raise ValueError((program, rows, cols))

        stage_start = arr(env.frame()).copy()
        for col, symbol in zip(cols, program):
            for row_index, row in enumerate(rows):
                is_on = int(stage_start[row][col]) == 5
                should_be_on = row_index in codes[symbol]
                if is_on != should_be_on:
                    env.step(6, col, row)
        configured = arr(env.frame()).copy()
        submit_discs = tuple(
            (blob.bbox, blob.area, blob.centroid)
            for blob in connected_components(configured, colors=(9,), min_area=4)
            if blob.size[0] > 1 and blob.size[1] > 1
        )
        click_largest_color_9_submit_disc(env)
        final = arr(env.frame()).copy()
        print(
            "candidate",
            {
                "stage": stage_index,
                "program": program,
                "selection": selection,
                "level": env.levels_completed,
                "submit_discs": submit_discs,
                "config_pixels": frame_delta(stage_start, configured)["count"],
                "submit_delta": frame_delta(configured, final),
                "total_delta": frame_delta(stage_start, final),
                "transitions": transitions(configured, final),
                "colors": color_counts(final),
                "configured_cyan": cyan_cells(configured),
                "final_cyan": cyan_cells(final),
                "board_objects": board_objects(final),
            },
        )
        if env.levels_completed > 6:
            break


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
