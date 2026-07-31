"""Compact clean-room observations for the pristine level-5 entry."""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta


def load_solver():
    spec = importlib.util.spec_from_file_location("solve", "solve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reach_level_5(env):
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as handle:
            checkpoint = json.load(handle)
        if (
            checkpoint.get("game") == "ls20"
            and checkpoint.get("validated")
            and checkpoint.get("final_path")
        ):
            for action in checkpoint["final_path"]:
                env.step(action)
    load_solver().solve(env)


def summarize_delta(before, after):
    delta = frame_delta(before, after)
    transitions = {}
    a = np.asarray(before)
    b = np.asarray(after)
    for old, new in zip(a[a != b], b[a != b]):
        key = (int(old), int(new))
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "count": delta["count"],
        "bbox": delta["bbox"],
        "transitions": sorted(transitions.items()),
    }


def world_marker_boxes(frame):
    boxes = []
    for color in (0, 8, 9, 11, 12, 14):
        for blob in connected_components(frame, colors=(color,), min_area=1):
            r0, c0, r1, c1 = blob.bbox
            if r0 < 60 and c1 >= 4:
                boxes.append((color, blob.bbox, blob.area))
    return boxes


def tile_map(frame):
    frame = np.asarray(frame)
    symbols = {0: "0", 1: "|", 3: ".", 4: "#", 5: "5",
               8: "8", 9: "A", 11: "x", 12: "A", 14: "e"}
    rows = []
    signatures = {}
    for tile_r in range(12):
        row = []
        for tile_c in range(12):
            block = frame[tile_r * 5:(tile_r + 1) * 5,
                          4 + tile_c * 5:4 + (tile_c + 1) * 5]
            values, counts = np.unique(block, return_counts=True)
            present = {int(value) for value in values}
            special = present & {0, 8, 9, 11, 12, 14}
            if 9 in special or 12 in special:
                token = "A"
            elif special:
                token = "".join(symbols[value] for value in sorted(special))
            else:
                majority = int(values[int(np.argmax(counts))])
                token = symbols.get(majority, str(majority))
            row.append(f"{token:2}")
            signatures[(tile_r, tile_c)] = tuple(
                (int(value), int(count))
                for value, count in zip(values, counts)
            )
        rows.append(" ".join(row))
    return rows, signatures


def inspect(env):
    reach_level_5(env)
    print("entry", env.levels_completed, env.terminal(), "actions", env.actions)
    frame = np.asarray(env.frame()).copy()
    print("colors", color_counts(frame))
    rows, signatures = tile_map(frame)
    print("tile_map")
    print("\n".join(rows))
    print(
        "special_tiles",
        [
            (cell, signature)
            for cell, signature in signatures.items()
            if any(color in {0, 8, 9, 11, 12, 14} for color, _ in signature)
        ],
    )
    print("markers", world_marker_boxes(frame))
    blobs = connected_components(frame, min_area=4)
    print(
        "blobs",
        [
            (blob.color, blob.bbox, blob.area)
            for blob in blobs
            if blob.area < 2000
        ],
    )
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        print(
            "action",
            action,
            "level",
            clone.levels_completed,
            "markers",
            world_marker_boxes(clone.frame()),
            "delta",
            summarize_delta(frame, clone.frame()),
        )


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
