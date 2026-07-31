"""Probe coordinate-sensitive action 6 at symbolic level-2 objects."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def probe(env):
    play_level_1(env)
    base = perception.arr(env.frame()).copy()
    points = [(3, 3), (12, 12)]
    for blob in perception.connected_components(
        env.frame(), colors=(0, 8, 9, 11, 14), min_area=4
    ):
        points.append((int(round(blob.centroid[1])), int(round(blob.centroid[0]))))
    for x, y in points:
        child = env.clone()
        try:
            child.step(6, x, y)
            delta = perception.frame_delta(base, child.frame())
            print(
                "point", (x, y), "color", int(base[y, x]),
                "delta", (delta["count"], delta["bbox"]),
                "level", child.levels_completed,
            )
        except Exception as exc:
            print("point", (x, y), "error", type(exc).__name__, str(exc))


arena.run_program("cn04", probe)
