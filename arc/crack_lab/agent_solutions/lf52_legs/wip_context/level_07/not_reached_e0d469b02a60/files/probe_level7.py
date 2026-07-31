import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from perception import color_counts, connected_components, frame_delta


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.size, b.area)
        for b in connected_components(frame, min_area=4)
        if b.area < 1000
    ]


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    print("AT", env.levels_completed, "actions", tuple(env.actions))
    print("COLORS", color_counts(env.frame()))
    print("BLOBS", compact_blobs(env.frame()))
    base = env.frame()
    for action in (1, 2, 3, 4):
        node = env.clone()
        node.step(action)
        delta = frame_delta(base, node.frame())
        print("KEY", action, "level", node.levels_completed,
              "delta", (delta["count"], delta["bbox"]))


levels, path, err = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), err)
