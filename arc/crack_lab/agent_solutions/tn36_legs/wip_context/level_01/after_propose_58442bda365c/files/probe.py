import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
from perception import connected_components, frame_delta


POINTS = {
    "pawn": (31, 14),
    "ahead1": (31, 18),
    "ahead2": (31, 22),
    "leftdiag": (27, 18),
    "rightdiag": (35, 18),
    "king": (31, 35),
    "empty": (15, 10),
}


def cyan(frame):
    return [(b.bbox, b.area) for b in connected_components(frame, colors=[11])]


def summarize(env):
    base = np.asarray(env.frame()).copy()
    for label, point in POINTS.items():
        clone = env.clone()
        clone.step(6, *point)
        after = np.asarray(clone.frame()).copy()
        d = frame_delta(base, after)
        print(label, point, "reward", clone.levels_completed,
              "cyan", cyan(after), "delta", d["count"], d["bbox"])
    for labels in [
        ("pawn", "ahead1"), ("pawn", "ahead2"),
        ("ahead1", "ahead1"), ("ahead2", "ahead2"),
        ("pawn", "leftdiag"), ("pawn", "rightdiag"),
        ("pawn", "pawn"), ("pawn", "king"),
    ]:
        clone = env.clone()
        frames = []
        for label in labels:
            clone.step(6, *POINTS[label])
            frames.append(cyan(np.asarray(clone.frame()).copy()))
        print("seq", labels, "reward", clone.levels_completed, "cyan_steps", frames)


A.run_program("tn36", summarize)
