import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board
from perception import color_counts, connected_components, frame_delta


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def small_blobs(frame):
    return tuple(
        (blob.color, blob.bbox, blob.size, blob.area)
        for blob in connected_components(frame, min_area=4)
        if blob.area < 100
    )


def probe(env):
    with open("checkpoint.json") as stream:
        actions = json.load(stream)["final_path"]
    for action in actions:
        env.step(*action) if isinstance(action, list) else env.step(action)

    base = env.frame()
    print("ENTRY", env.levels_completed, tuple(env.actions))
    print("COLORS", color_counts(base))
    print("BOARD", board(base))
    print("BLOBS", small_blobs(base))
    for action in (1, 2, 3, 4):
        node = env.clone()
        node.step(action)
        delta = frame_delta(base, node.frame())
        print("KEY", action, (delta["count"], delta["bbox"]), board(node.frame()))
    for row, column in ((1, 1), (12, 6), (12, 42), (42, 12), (42, 42)):
        node = env.clone()
        node.step(6, column + 1, row + 1)
        delta = frame_delta(base, node.frame())
        print("CLICK", (row, column), (delta["count"], delta["bbox"]),
              board(node.frame()))


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
