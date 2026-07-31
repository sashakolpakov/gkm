import json

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups, body_pixels


SUFFIX_23 = [
    (6, 54, 53), (6, 7, 51), (6, 51, 19), (6, 54, 53),
    (6, 0, 54), (6, 55, 19), (6, 3, 51), (6, 50, 49),
]


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_masks = tuple(
        {
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        }
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def overlap(groups):
        pixels = {point for group in groups for point in group}
        return sum(point in mask for point in pixels for mask in ring_masks)

    print("TRACE", 0, overlap(body_groups(root.frame())))
    for index, action in enumerate(SUFFIX_23, 1):
        root.step(*action)
        groups = body_groups(root.frame())
        print(
            "TRACE",
            index,
            overlap(groups),
            tuple(
                (
                    round(sum(row for row, _ in group) / len(group)),
                    round(sum(col for _, col in group) / len(group)),
                )
                for group in groups
            ),
        )

    groups = body_groups(root.frame())
    print("ROOT23", int(root.levels_completed), "overlap", overlap(groups))
    for group in groups:
        row0 = min(row for row, _ in group)
        col0 = min(col for _, col in group)
        print(
            "BODY",
            (row0, col0),
            tuple((row - row0, col - col0) for row, col in group),
            "rings",
            tuple(len(set(group) & mask) for mask in ring_masks),
        )

    outcomes = {}
    actions = [
        (6, col, row) for row, col in sorted(body_pixels(root.frame()))
    ]
    actions.append((6, 32, 32))
    for action in actions:
        child = root.clone()
        child.step(*action)
        groups = body_groups(child.frame())
        key = groups
        outcomes.setdefault(key, []).append(action)
        if int(child.levels_completed) > start_level:
            print("FOUND_NEXT", action)
            return
    ranked = sorted(
        outcomes.items(), key=lambda item: -overlap(item[0])
    )
    print("OUTCOMES", len(outcomes))
    for groups, result_actions in ranked[:12]:
        print(
            "OUT",
            overlap(groups),
            tuple(
                (
                    round(sum(row for row, _ in group) / len(group)),
                    round(sum(col for _, col in group) / len(group)),
                )
                for group in groups
            ),
            result_actions,
        )


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
