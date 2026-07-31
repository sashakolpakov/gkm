import gkm_try as harness
import perception as p
import numpy as np


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(frame, min_area=4)
    ]


def row_segments(frame, row, background=5):
    values = np.asarray(frame)[row]
    out = []
    start = 0
    for col in range(1, len(values) + 1):
        if col == len(values) or values[col] != values[start]:
            color = int(values[start])
            if color != background:
                out.append((start, col - 1, color))
            start = col
    return out


def transition_counts(before, after):
    a, b = np.asarray(before), np.asarray(after)
    pairs, counts = np.unique(
        np.stack((a[a != b], b[a != b]), axis=1), axis=0, return_counts=True
    )
    return [(int(x), int(y), int(n)) for (x, y), n in zip(pairs, counts)]


def observe(env):
    harness.m.solve(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", p.color_counts(env.frame()))
    print("objects", compact_objects(env.frame()))
    for row in range(64):
        segments = row_segments(env.frame(), row)
        if segments:
            print("row", row, segments)
    base = np.asarray(env.frame()).copy()
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        delta = p.frame_delta(base, clone.frame())
        print(
            "action", action, "delta", delta["count"], delta["bbox"],
            "transitions", transition_counts(base, clone.frame())
        )


harness.A.run_program("cd82", observe)
