"""Compact clean-room observations for the pristine sk48 level-8 entry."""

import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def compact_objects(frame):
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(frame, min_area=4)
    ]


def compact_delta(before, after):
    delta = frame_delta(before, after)
    counts = {}
    for row, col, old, new in delta["samples"]:
        key = (old, new)
        counts[key] = counts.get(key, 0) + 1
    return delta["count"], delta["bbox"], sorted(counts.items())


def active_state(frame):
    blobs = connected_components(frame, min_area=4)
    heads = [
        (b.color, tuple(round(v, 1) for v in b.centroid))
        for b in blobs
        if b.color in (6, 15) and b.area >= 18 and b.centroid[0] < 53
    ]
    tokens = [
        (b.color, tuple(round(v, 1) for v in b.centroid))
        for b in blobs
        if b.color in (8, 9, 12, 14) and b.area == 16 and b.centroid[0] < 53
    ]
    return heads, tokens


def lattice(frame):
    rows = []
    for grid_row in range(8):
        cells = []
        for grid_col in range(8):
            row = 2 + 6 * grid_row
            col = 5 + 6 * grid_col
            values = sorted(
                {
                    int(value)
                    for value in frame[row : row + 6, col : col + 6].flat
                    if int(value) not in (4, 5)
                }
            )
            cells.append("".join(f"{value:x}" for value in values) or ".")
        rows.append(" ".join(f"{cell:>4}" for cell in cells))
    return rows


def probe(env):
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    else:
        checkpoint = None
    if (
        checkpoint
        and checkpoint.get("game") == "sk48"
        and checkpoint.get("validated")
        and checkpoint.get("final_path")
    ):
        for action in checkpoint["final_path"]:
            env.step(action)
    else:
        solver.solve(env)
    base = env.frame()
    print("ENTRY", env.levels_completed, "actions", env.actions)
    print("COUNTS", color_counts(base))
    print("OBJECTS", compact_objects(base))
    print("LATTICE")
    print("\n".join(lattice(base)))

    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        print("KEY", action, compact_delta(base, clone.frame()), active_state(clone.frame()))

    selectable = [
        b for b in connected_components(base, min_area=16)
        if b.color not in (1, 2)
    ]
    for blob in selectable:
        clone = env.clone()
        row, col = blob.centroid
        clone.step(6, round(col), round(row))
        delta = compact_delta(base, clone.frame())
        if delta[0]:
            print(
                "CLICK",
                (blob.color, blob.bbox, round(col), round(row)),
                delta,
            )
            selected = clone.frame()
            for action in (1, 2, 3, 4):
                child = clone.clone()
                child.step(action)
                print(
                    "AFTER_CLICK_KEY",
                    blob.color,
                    action,
                    compact_delta(selected, child.frame()),
                    active_state(child.frame()),
                )


levels, path, err = arena.run_program("sk48", probe)
print("PROBE_RESULT", levels, len(path), err)
