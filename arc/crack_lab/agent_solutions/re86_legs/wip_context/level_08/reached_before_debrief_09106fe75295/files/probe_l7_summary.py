import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, color_counts, connected_components
from probe_l8_routes import selected_shape
from search_l7_segments import PREFIX, SEGMENTS, cursor, expand, step_many


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    step_many(env, full[:PREFIX])
    root_frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(root_frame, colors=(4,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    centers = [(blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in rings]
    targets = [(point, int(root_frame[point])) for point in centers]
    print("targets", targets)
    print("root_cursor", cursor(root_frame), "shape", selected_shape(env))
    print("root_counts", color_counts(root_frame))
    for index, runs in enumerate(SEGMENTS, 1):
        step_many(env, expand(runs))
        frame = arr(env.frame())
        print(
            "stage",
            index,
            "cursor",
            cursor(frame),
            "values",
            [int(frame[point]) for point in centers],
            "shape",
            selected_shape(env),
            "counts",
            color_counts(frame),
        )
        if index < len(SEGMENTS):
            env.step(5)


if __name__ == "__main__":
    A.run_program("re86", probe)
