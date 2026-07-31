import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=2)
    ]


def probe(env):
    print("actions", env.actions)
    base = env.frame()
    print("counts", color_counts(base))
    print("blobs", compact_blobs(base))
    for action in env.actions:
        clone = env.clone()
        try:
            clone.step(action)
            print("key", action, frame_delta(base, clone.frame()))
        except Exception as exc:
            print("key", action, type(exc).__name__, str(exc))
    points = [
        (4, 59),    # lower color-15 object
        (10, 53),   # small color-0 object
        (48, 15),   # color-3/9 object
        (30, 30),   # empty center
        (60, 55),   # empty lower-right
    ]
    for x, y in points:
        clone = env.clone()
        try:
            clone.step(6, x, y)
            print("click", (x, y), frame_delta(base, clone.frame()))
        except Exception as exc:
            print("click", (x, y), type(exc).__name__, str(exc))
    sequences = {
        "black_target": [(10, 53), (48, 15)],
        "black_ring": [(10, 53), (44, 15)],
        "black_center": [(10, 53), (30, 30)],
        "target_black": [(48, 15), (10, 53)],
        "diag2": [(10, 53), (16, 47)],
        "diag4": [(10, 53), (16, 47), (22, 41), (28, 35)],
        "diag_goal": [
            (10, 53), (16, 47), (22, 41), (28, 35),
            (34, 29), (40, 23), (46, 17), (48, 15),
        ],
    }
    for name, seq in sequences.items():
        clone = env.clone()
        before = clone.frame()
        for x, y in seq:
            clone.step(6, x, y)
        print(
            "seq", name,
            "level", clone.levels_completed,
            "counts", color_counts(clone.frame()),
            "delta", frame_delta(before, clone.frame()),
            "blobs", compact_blobs(clone.frame()),
        )


A.run_program("su15", probe)
