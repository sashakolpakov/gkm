import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import follow_diagonal_lattice_to_ring
from perception import color_counts, connected_components, frame_delta


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(frame, min_area=3)
        if b.color != 4
    ]


def small_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame)
        if b.area <= 100
    ]


def probe(env):
    follow_diagonal_lattice_to_ring(env)
    base = env.frame()
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(base))
    print("blobs", compact_blobs(base))

    points = {(0, 0), (32, 32), (63, 63)}
    for blob in connected_components(base, min_area=3):
        if blob.color != 4:
            r, c = blob.centroid
            points.add((int(round(c)), int(round(r))))
    for action in env.actions:
        for x, y in sorted(points):
            clone = env.clone()
            try:
                clone.step(int(action), x, y)
            except TypeError:
                clone.step(int(action))
            delta = frame_delta(base, clone.frame())
            if delta["count"] or clone.levels_completed != env.levels_completed:
                print(
                    "try", (int(action), x, y),
                    "delta", (delta["count"], delta["bbox"]),
                    "cells", delta["samples"],
                    "level", clone.levels_completed,
                )
    for point in ((0, 0), (33, 27), (31, 36), (32, 50), (10, 20), (55, 20)):
        clone = env.clone()
        print("seq_start", point)
        for n in range(1, 13):
            before = clone.frame()
            clone.step(6, *point)
            delta = frame_delta(before, clone.frame())
            print(
                "seq", n, "d", (delta["count"], delta["bbox"]),
                "small", small_blobs(clone.frame()),
                "level", clone.levels_completed,
            )
            if clone.levels_completed != env.levels_completed:
                break
    for path in (
        ((33, 34), (33, 28)),
        ((33, 34), (39, 34), (41, 37)),
        ((31, 36), (27, 37), (23, 37), (19, 37)),
    ):
        clone = env.clone()
        print("path_start", path)
        for point in path:
            before = clone.frame()
            clone.step(6, *point)
            delta = frame_delta(before, clone.frame())
            print(
                "path_step", point, "cells", delta["samples"],
                "level", clone.levels_completed,
            )
            if clone.levels_completed != env.levels_completed:
                break


levels, path, err = A.run_program("su15", probe)
print("probe_result", levels, len(path), err)
