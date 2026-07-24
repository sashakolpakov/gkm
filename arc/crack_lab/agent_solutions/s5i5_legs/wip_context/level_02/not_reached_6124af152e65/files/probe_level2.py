import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import align_slider_tips_to_hollow_targets
from perception import arr, color_counts, frame_delta, object_candidates


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"], tuple(round(v, 1) for v in o["centroid"]))
        for o in object_candidates(frame)
    ]


def active_parts(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame)
        if o["color"] in (10, 11, 12, 14) and o["bbox"][0] < 50
    ]


def marker(frame):
    grid = arr(frame)
    return [
        (int(r), int(c))
        for r, c in zip(*((grid == 13).nonzero()))
        if not (30 <= r <= 32 and 49 <= c <= 51)
    ]


def probe(env):
    align_slider_tips_to_hollow_targets(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    base = env.frame()
    print("COLORS", color_counts(base))
    print("OBJECTS", compact_objects(base))
    print("MARKER", marker(base))
    symbols = {5: ".", 15: "#", 13: "G", 10: "A", 11: "B", 12: "C", 14: "D"}
    for r, row in enumerate(base[:54]):
        line = "".join(symbols.get(int(v), "?") for v in row)
        if set(line) != {"."}:
            print(f"MAP {r:02d} {line}")
    for color, bbox, area, centroid in compact_objects(base):
        y, x = centroid
        clone = env.clone()
        clone.step(6, int(round(x)), int(round(y)))
        delta = frame_delta(base, clone.frame())
        if delta["count"]:
            print(
                "CLICK",
                (color, bbox, int(round(x)), int(round(y))),
                "DELTA",
                (delta["count"], delta["bbox"]),
                "LEVEL",
                clone.levels_completed,
                "COLORS",
                color_counts(clone.frame()),
            )
    for x in (12, 27, 42, 57):
        clone = env.clone()
        print("SERIES", x, "START", active_parts(clone.frame()))
        for n in range(1, 9):
            before = clone.frame()
            clone.step(6, x, 57)
            delta = frame_delta(before, clone.frame())
            print(
                "STEP",
                n,
                (delta["count"], delta["bbox"]),
                "LEVEL",
                clone.levels_completed,
                "PARTS",
                active_parts(clone.frame()),
            )
            if not delta["count"] or clone.levels_completed > 1:
                break
    for left, right in ((6, 12), (21, 27), (36, 42), (51, 57)):
        for sequence in ((left,), (right,), (right, left), (right, right, left)):
            clone = env.clone()
            states = [marker(clone.frame())]
            for x in sequence:
                clone.step(6, x, 57)
                states.append(marker(clone.frame()))
            print("CONTROL", sequence, "MARKERS", states)


A.run_program("s5i5", probe)
