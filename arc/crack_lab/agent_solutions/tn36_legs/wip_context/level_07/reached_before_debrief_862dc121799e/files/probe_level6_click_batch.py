"""Map selected coordinate clicks in independent fresh level-6 Arenas."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components, frame_delta


with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)


def shape(frame, color, left_half):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(color,), min_area=4)
        if blob.bbox[2] < 32 and ((blob.bbox[1] < 32) == left_half)
    ]
    if not blobs:
        return ()
    blob = max(blobs, key=lambda item: item.area)
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row][col]) == color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def run(col, row):
    observation = {}

    def observe(env):
        for action in CHECKPOINT["final_path"]:
            env.step(action)
        before = arr(env.frame()).copy()
        env.step(6, col, row)
        after = arr(env.frame()).copy()
        observation.update(
            {
                "at": (col, row),
                "pixel": int(before[row][col]),
                "delta": frame_delta(before, after),
                "yellow": shape(after, 4, True),
                "cyan": shape(after, 11, False),
                "level": env.levels_completed,
            }
        )

    A.run_program("tn36", observe)
    return observation


for value in sys.argv[1:]:
    x_text, y_text = value.split(",")
    print("click", run(int(x_text), int(y_text)))
