import json

import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups


PATH = (
    (6, 7, 48),
    (6, 7, 54),
    (6, 15, 19),
    (6, 7, 19),
    (6, 16, 53),
    (6, 31, 44),
    (6, 42, 43),
    (6, 10, 53),
    (6, 7, 55),
)


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    rings = tuple(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )

    def summary(node):
        frame = node.frame()
        solids = tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(frame, min_area=1)
            if (
                blob.bbox[0] >= 10
                and blob.size[0] == blob.size[1]
                and blob.area == blob.size[0] ** 2
                and blob.color not in (3, 4, 5, 7, 9)
            )
        )
        movers = tuple(
            (color, center(group), len(group))
            for color in (7, 14, 13)
            for group in groups(frame, color)
        )
        status = tuple(
            (color, sum(
                int(frame[row][col]) == color
                for row in range(10)
                for col in range(24, 64)
            ))
            for color in (12, 14)
        )
        return solids, movers, status

    print("ROOT", summary(env), "rings", rings)
    for index, action in enumerate(PATH, 1):
        env.step(*action)
        print(
            "STEP", index, action, summary(env),
            "level", int(env.levels_completed),
        )
        if env.terminal() or int(env.levels_completed) > start_level:
            return


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
