"""Verify a compact three-platform relay candidate for vc33 level 7."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed

    def summary():
        frame = arr(env.frame())
        fluid = (
            (frame == 3)
            | (frame == 11)
            | (frame == 14)
            | (frame == 15)
        )

        def top(rows, cols):
            points = fluid[rows, cols].nonzero()[0]
            return None if len(points) == 0 else rows.start + int(points.min())

        surfaces = (
            top(slice(8, 30), slice(8, 22)),
            top(slice(8, 56), slice(24, 40)),
            top(slice(8, 30), slice(42, 56)),
            top(slice(30, 56), slice(8, 22)),
            top(slice(30, 56), slice(42, 56)),
        )
        markers = {}
        for color in (11, 14, 15):
            markers[color] = [
                (blob.area, tuple(round(value, 1) for value in blob.centroid))
                for blob in connected_components(
                    env.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
        gates = [
            (blob.color, blob.bbox)
            for blob in connected_components(
                env.frame(), colors=(1, 12, 13, 14, 15), min_area=8
            )
        ]
        return env.levels_completed - start_level, surfaces, markers, gates

    groups = [
        (
            "align15",
            [(6, 24, 8), (6, 42, 8), (6, 40, 19)]
            + [(6, 24, 8)] * 9
            + [(6, 38, 8)] * 4
            + [(6, 24, 32)] * 4,
        ),
        (
            "cross11",
            [(6, 20, 8)] * 3
            + [(6, 20, 32)] * 3
            + [(6, 22, 41)]
            + [(6, 20, 8)] * 6,
        ),
        (
            "ready11_right",
            [(6, 20, 8)] + [(6, 42, 8)] * 4,
        ),
        ("cross11_right", [(6, 40, 19)]),
        (
            "ready14",
            [(6, 24, 8)] * 9 + [(6, 38, 32)] * 2,
        ),
        ("cross14", [(6, 40, 41)]),
        (
            "finish",
            [(6, 38, 32)] + [(6, 20, 32)] * 6 + [(6, 42, 8)] * 5,
        ),
    ]
    path = []
    for label, actions in groups:
        for action in actions:
            if env.levels_completed > start_level:
                break
            env.step(*action)
            path.append(action)
        print(label, len(path), summary())
        if env.levels_completed > start_level:
            break
    print("result", env.levels_completed - start_level, len(path), path)


arena.run_program("vc33", probe)
