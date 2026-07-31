"""Verify the union-of-context protocol target on a pristine clone."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from perception import connected_components


ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLUMNS = (8, 13, 18, 23)
RIGHT_COLUMNS = (34, 39, 44, 49, 54, 59)


def active_rows(frame, columns):
    return {
        index
        for index, row in enumerate(ROWS)
        if any(int(frame[row, column]) == 5 for column in columns)
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    context_rows = {}
    context_probe = env.clone()
    context_rows["D"] = active_rows(np.asarray(context_probe.frame()), LEFT_COLUMNS)
    for name, x in (("U", 15), ("L", 25), ("R", 35)):
        context_probe.step(6, x, 58)
        context_rows[name] = active_rows(np.asarray(context_probe.frame()), LEFT_COLUMNS)

    print("contexts", {name: sorted(rows) for name, rows in context_rows.items()})
    for name, target_rows in context_rows.items():
        clone = env.clone()
        before = np.asarray(clone.frame())
        toggles = [
            (6, column, ROWS[row_index])
            for row_index in sorted(target_rows)
            for column in RIGHT_COLUMNS
            if int(before[ROWS[row_index], column]) == 1
        ]
        for action in toggles:
            clone.step(*action)

        submit = max(
            (
                blob
                for blob in connected_components(clone.frame(), colors=(9,), min_area=4)
                if blob.size[0] > 1 and blob.size[1] > 1
            ),
            key=lambda blob: blob.area,
        )
        row, column = submit.centroid
        clone.step(6, int(round(column)), int(round(row)))
        print(
            "candidate",
            {
                "name": name,
                "rows": sorted(target_rows),
                "toggles": len(toggles),
                "levels_after_submit": clone.levels_completed,
                "terminal": clone.terminal(),
            },
        )

    protocols = (
        "URRRUR",
        "URRRRU",
        "RURRUR",
        "RURRRU",
        "DLLLDL",
        "DLLLLD",
        "LDLLDL",
        "LDLLLD",
    )
    for protocol in protocols:
        clone = env.clone()
        before = np.asarray(clone.frame())
        toggles = [
            (6, RIGHT_COLUMNS[column_index], ROWS[row_index])
            for column_index, direction in enumerate(protocol)
            for row_index in sorted(context_rows[direction])
            if int(before[ROWS[row_index], RIGHT_COLUMNS[column_index]]) == 1
        ]
        for action in toggles:
            clone.step(*action)
        submit = max(
            (
                blob
                for blob in connected_components(clone.frame(), colors=(9,), min_area=4)
                if blob.size[0] > 1 and blob.size[1] > 1
            ),
            key=lambda blob: blob.area,
        )
        row, column = submit.centroid
        clone.step(6, int(round(column)), int(round(row)))
        print(
            "protocol",
            {
                "directions": protocol,
                "toggles": len(toggles),
                "levels_after_submit": clone.levels_completed,
                "terminal": clone.terminal(),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
