"""Compact clean-room observations for sb26 level 5."""

import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import action_deltas, color_counts, connected_components, frame_delta


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solve_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve_module)


def summarize(env):
    frame = env.frame()
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("counts", color_counts(frame))
    blobs = connected_components(frame, min_area=4)
    print(
        "blobs",
        [
            (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
            for b in blobs
        ],
    )
    print(
        "keys",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    for blob in blobs:
        x = int(round(blob.centroid[1]))
        y = int(round(blob.centroid[0]))
        clone = env.clone()
        before = clone.frame().copy()
        clone.step(6, x, y)
        delta = frame_delta(before, clone.frame())
        if delta["count"] or clone.levels_completed != env.levels_completed:
            print(
                "click",
                (blob.color, blob.bbox, x, y),
                "delta",
                (delta["count"], delta["bbox"]),
                "level",
                clone.levels_completed,
            )

    choices = [
        (6, 58),
        (14, 58),
        (20, 58),
        (28, 58),
        (34, 58),
        (42, 58),
        (48, 58),
        (56, 58),
    ]
    for choice in choices:
        clone = env.clone()
        clone.step(6, *choice)
        selected = frame_delta(frame, clone.frame())
        clone.step(5)
        print(
            "choose-submit",
            choice,
            "select_delta",
            (selected["count"], selected["bbox"]),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
        )
    slots = [(20, 23), (26, 23), (32, 23), (38, 23), (44, 23),
             (26, 37), (32, 37), (38, 37)]
    for choice in choices:
        for slot in slots:
            clone = env.clone()
            clone.step(6, *choice)
            before_slot = clone.frame().copy()
            clone.step(6, *slot)
            delta = frame_delta(before_slot, clone.frame())
            if delta["count"]:
                central = [
                    sample for sample in delta["samples"]
                    if 16 <= sample[0] <= 42
                ]
                print(
                    "paint",
                    choice,
                    slot,
                    (delta["count"], delta["bbox"]),
                    central,
                )


def probe(env):
    solve_module.solve(env)
    summarize(env)


levels, path, err = A.run_program("sb26", probe)
print("done", levels, len(path), err)
