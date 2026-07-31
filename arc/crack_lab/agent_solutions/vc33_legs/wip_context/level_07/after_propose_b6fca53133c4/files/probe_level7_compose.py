"""Test existing horizontal-gate leg one marker at a time on level 7."""
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import cross_horizontal_gates_then_align_opposing_markers
from perception import connected_components


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start = env.levels_completed

    def gaps():
        out = {}
        for color in (11, 14, 15):
            blobs = [
                blob
                for blob in connected_components(
                    env.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
            if len(blobs) == 2:
                moving = max(blobs, key=lambda blob: blob.area)
                fixed = min(blobs, key=lambda blob: blob.area)
                out[color] = (
                    abs(moving.centroid[0] - fixed.centroid[0]),
                    abs(moving.centroid[1] - fixed.centroid[1]),
                )
        return out

    print("start", gaps())
    for color in (15, 14, 11):
        cross_horizontal_gates_then_align_opposing_markers(
            env,
            marker_colors=(color,),
            max_stages=32,
            max_states=3000,
            max_depth=28,
        )
        print("stage", color, env.levels_completed, gaps())
        if env.levels_completed > start:
            break


arena.run_program("vc33", probe)
