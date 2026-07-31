import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


SYMBOLS = {0: "s", 1: ".", 2: "#", 4: "t", 5: "s",
           11: "G", 12: "a", 13: "A", 14: "o", 15: "|"}


def crop(frame, bbox, margin=1):
    y0, x0, y1, x1 = bbox
    y0, x0 = max(0, y0 - margin), max(0, x0 - margin)
    y1, x1 = min(62, y1 + margin), min(62, x1 + margin)
    return tuple(
        "".join(SYMBOLS[int(value)] for value in frame[y, x0:x1 + 1])
        for y in range(y0, y1 + 1)
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    frame = arr(env.frame())[:63]
    for blob in connected_components(
        frame, colors=(4, 11, 13, 14), min_area=8
    ):
        print(blob.color, blob.bbox, blob.area, crop(frame, blob.bbox))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
