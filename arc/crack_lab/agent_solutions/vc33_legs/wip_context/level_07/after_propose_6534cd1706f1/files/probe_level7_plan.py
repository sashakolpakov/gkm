"""Test a conservation-based three-reservoir solution for level 7."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed

    def marker_rows():
        result = {}
        for color in (11, 14, 15):
            result[color] = [
                round(blob.centroid[0], 1)
                for blob in connected_components(
                    env.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
        return result

    cross_and_align_15 = [
        (6, 24, 8),
        (6, 42, 8),
        (6, 40, 19),
        *([(6, 24, 8)] * 9),
        *([(6, 38, 8)] * 4),
        *([(6, 24, 32)] * 4),
    ]
    for action in cross_and_align_15:
        env.step(*action)
    print("aligned15", env.levels_completed, marker_rows())

    for _ in range(3):
        env.step(6, 24, 32)
        env.step(6, 20, 8)
    print("aligned11", env.levels_completed, marker_rows())

    upper_refills = [(6, 20, 8)] * 7 + [(6, 42, 8)] * 5
    for refill in upper_refills:
        env.step(6, 38, 32)
        env.step(*refill)
        if env.levels_completed > start_level:
            break
    print("result", env.levels_completed, marker_rows())


arena.run_program("vc33", probe)
