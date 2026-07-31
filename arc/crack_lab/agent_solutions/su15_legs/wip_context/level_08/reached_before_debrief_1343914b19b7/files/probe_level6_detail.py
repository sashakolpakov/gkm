import gkm_try as H

import players
from perception import connected_components
from probe_clean8 import body_groups


PATH = (
    (6, 28, 27), (6, 20, 19), (6, 12, 12), (6, 6, 16),
    (6, 56, 57), (6, 56, 57), (6, 56, 57),
)


def inspect(env):
    while env.levels_completed < 5:
        level = int(env.levels_completed) + 1
        getattr(players, f"play_level_{level}")(env)
    start = int(env.levels_completed)
    initial = env.frame()
    symbols = "0123456789ABCDEF"
    for row in range(10):
        print("UI", row, "".join(symbols[int(initial[row][col])]
                                 for col in range(64)))
    ring_mask = {
        (row, col)
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
        for row in range(blob.bbox[0], blob.bbox[2] + 1)
        for col in range(blob.bbox[1], blob.bbox[3] + 1)
        if int(initial[row][col]) == 9
    }

    def state():
        frame = env.frame()
        groups = body_groups(frame)
        body_overlap = len(
            {point for group in groups for point in group} & ring_mask
        )
        squares = tuple(
            (blob.color, blob.bbox)
            for blob in connected_components(frame, min_area=4)
            if (
                blob.bbox[0] >= 10
                and blob.size[0] == blob.size[1]
                and blob.area == blob.size[0] ** 2
                and blob.color not in (3, 4, 5, 7, 9)
            )
        )
        return squares, tuple(
            (
                round(sum(row for row, _ in group) / len(group)),
                round(sum(col for _, col in group) / len(group)),
                len(group),
            )
            for group in groups
        ), body_overlap

    print("ROOT", state())
    for index, action in enumerate(PATH, 1):
        before = state()
        env.step(*action)
        print(index, action, "before", before, "after", state(),
              "level", int(env.levels_completed))
        if int(env.levels_completed) > start:
            return


H.A.run_program("su15", inspect)
