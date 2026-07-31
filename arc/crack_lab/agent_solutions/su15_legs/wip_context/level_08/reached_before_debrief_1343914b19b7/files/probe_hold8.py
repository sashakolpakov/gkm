import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_mask = {
        (row, col)
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
        for row in range(blob.bbox[0], blob.bbox[2] + 1)
        for col in range(blob.bbox[1], blob.bbox[3] + 1)
        if int(initial[row][col]) == 9
    }
    for action in PREFIX:
        env.step(*action)
    groups = body_groups(env.frame())
    actions = [
        (6, col, row)
        for group in groups
        for row, col in tuple(group) + (center(group),)
    ]
    actions.append((6, 56, 15))
    outcomes = []
    for action in actions:
        child = env.clone()
        child.step(*action)
        child_groups = body_groups(child.frame())
        overlap = len(
            {point for group in child_groups for point in group} & ring_mask
        )
        outcomes.append(
            (
                overlap,
                int(child.levels_completed),
                tuple(center(group) for group in child_groups),
                action,
            )
        )
    for outcome in sorted(outcomes, reverse=True)[:15]:
        print(outcome)
        if outcome[1] > start_level:
            break


A.run_program("su15", inspect)
