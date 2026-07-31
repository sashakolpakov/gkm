"""Compact, observational reproduction of the preserved level-9 candidate."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta
from legs import (
    _bridge_carrier_state,
    _movable_bridge_board,
    solve_multi_bridge_wrapped_carrier_peg_solitaire,
)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def compact_components(frame):
    out = {}
    for color in (5, 9, 10, 14):
        blobs = connected_components(frame, colors=(color,), min_area=4)
        out[color] = {
            "count": len(blobs),
            "areas": sorted(blob.area for blob in blobs),
            "boxes": [blob.bbox for blob in blobs if blob.area <= 64],
        }
    return out


def lattice_signature(frame):
    array = np.asarray(frame)
    out = {}
    for row in range(0, 64, 6):
        for col in range(0, 64, 6):
            patch = array[row:row + 4, col:col + 4]
            values, counts = np.unique(patch, return_counts=True)
            signature = tuple((int(value), int(count)) for value, count in zip(values, counts))
            if signature != ((10, 16),):
                out[(row, col)] = signature
    return out


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    transitions = []
    previous_level = env.levels_completed
    for index, action in enumerate(checkpoint["final_path"], 1):
        env.step(action)
        if env.levels_completed != previous_level:
            transitions.append((env.levels_completed, index))
            previous_level = env.levels_completed
    print("PREFIX", {"length": len(checkpoint["final_path"]), "transitions": transitions})

    entry = env.clone()
    print("PARSED", {
        "movable": tuple(sorted(part) for part in _movable_bridge_board(entry.frame())),
        "bridge_carrier": _bridge_carrier_state(entry.frame()),
    })
    print("LATTICE", lattice_signature(entry.frame()))
    print("ENTRY", {
        "levels": entry.levels_completed,
        "actions": entry.actions,
        "colors": color_counts(entry.frame()),
        "components": compact_components(entry.frame()),
    })

    for action in (1, 2, 3, 4, 7):
        clone = entry.clone()
        before = clone.frame()
        clone.step(action)
        print("KEY", action, {
            "delta": frame_delta(before, clone.frame()),
            "levels": clone.levels_completed,
        })

    for point in ((32, 32), (19, 43), (31, 43)):
        clone = entry.clone()
        before = clone.frame()
        clone.step(6, *point)
        print("CLICK", point, {
            "delta": frame_delta(before, clone.frame()),
            "levels": clone.levels_completed,
        })

    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    leg_clone = entry.clone()
    solve_multi_bridge_wrapped_carrier_peg_solitaire(leg_clone)
    clone = entry.clone()
    milestones = []
    previous_counts = color_counts(clone.frame())
    for index, action in enumerate(candidate, 1):
        before = clone.frame()
        play(clone, action)
        counts = color_counts(clone.frame())
        if counts != previous_counts or clone.levels_completed != 8:
            milestones.append((
                index,
                action,
                clone.levels_completed,
                {color: counts.get(color, 0) for color in (5, 9, 10, 14)},
                frame_delta(before, clone.frame())["count"],
            ))
        previous_counts = counts
        if clone.levels_completed > 8:
            break
    print("LEG", {
        "levels": leg_clone.levels_completed,
        "matches_candidate_final": np.array_equal(leg_clone.frame(), clone.frame()),
    })
    print("CANDIDATE", {
        "declared_len": len(candidate),
        "executed": index,
        "levels": clone.levels_completed,
        "terminal": clone.terminal(),
        "milestones": milestones,
        "final_components": compact_components(clone.frame()),
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
