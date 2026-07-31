import importlib.util
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from legs import merge_equal_squares_around_moving_cutter


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def summarize(env):
    blobs = connected_components(env.frame(), min_area=1)
    compact = [
        (b.color, b.bbox, b.area)
        for b in blobs
        if b.color != 3 or b.area < 3000
    ]
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("blobs", compact)


def inspect(env):
    solver.solve(env)
    summarize(env)
    attempt = env.clone()
    merge_equal_squares_around_moving_cutter(
        attempt, max_states=5000, max_depth=36
    )
    print(
        "existing_leg", attempt.levels_completed,
        color_counts(attempt.frame()),
    )
    before = env.frame()
    points = [
        (32, 15), (7, 26), (43, 27), (54, 27), (15, 29),
        (44, 53), (14, 54), (58, 59), (3, 60),
        (6, 39), (48, 39), (5, 15), (58, 15), (32, 58),
    ]
    for x, y in points:
        clone = env.clone()
        try:
            clone.step(6, x, y)
            delta = frame_delta(before, clone.frame())
            moved = [
                (b.color, b.bbox, b.area)
                for b in connected_components(clone.frame(), min_area=1)
                if b.bbox[0] >= 10 and b.color in (6, 7, 9, 10, 11, 15)
            ]
            print(
                "click", (x, y), "delta", (delta["count"], delta["bbox"]),
                "level", clone.levels_completed, "objects", moved,
            )
        except Exception as exc:
            print("click", (x, y), "error", type(exc).__name__)


A.run_program("su15", inspect)
