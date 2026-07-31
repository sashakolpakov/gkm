import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups
from probe_verify_path8 import PATH


def inspect(env):
    H.resumed_solve(env)
    for index, action in enumerate(PATH, 1):
        _, col, row = action
        frame = env.frame()
        role = None
        for color in (7, 14):
            for group in groups(frame, color):
                if (row, col) in group:
                    point = center(group)
                    role = (
                        color,
                        point,
                        (row - point[0], col - point[1]),
                    )
        squares = tuple(
            (
                blob.color,
                (round(blob.centroid[0]), round(blob.centroid[1])),
                max(
                    abs(row - round(blob.centroid[0])),
                    abs(col - round(blob.centroid[1])),
                ),
            )
            for blob in connected_components(frame, min_area=4)
            if (
                blob.bbox[0] >= 10
                and blob.color in (8, 11, 12)
                and blob.area == blob.size[0] * blob.size[1]
            )
        )
        print(
            index, action,
            "pixel", int(frame[row][col]),
            "agent", role,
            "squares", squares,
        )
        env.step(*action)


H.A.run_program("su15", inspect)
