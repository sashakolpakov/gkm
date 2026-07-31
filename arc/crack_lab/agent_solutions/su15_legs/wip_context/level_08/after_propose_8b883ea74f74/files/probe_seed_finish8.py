import json
import time

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


SEED = (
    (0, (1, -1)), (0, (0, 2)), (1, (1, 1)), (2, (1, 0)),
    (2, (1, 1)), (0, (0, 2)), (1, (1, 1)),
)


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def labeled(groups):
    groups = list(groups)
    top = min(groups, key=lambda group: center(group)[0])
    groups.remove(top)
    left = min(groups, key=lambda group: center(group)[1])
    groups.remove(left)
    return top, groups[0], left


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)

    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_mask = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(initial[row][col]) == 9
    }
    node = env.clone()
    for action in PREFIX:
        node.step(*action)

    def summary(candidate):
        frame = candidate.frame()
        groups = body_groups(frame)
        overlap = sum(
            point in ring_mask for group in groups for point in group
        )
        centers = tuple(sorted(center(group) for group in groups))
        square = tuple(
            blob.bbox
            for blob in connected_components(frame, colors=(8,), min_area=9)
            if blob.bbox[0] >= 10
        )
        return overlap, square, centers

    actions = []
    print("STAGED", summary(node))
    for index, (label, offset) in enumerate(SEED, 1):
        group = labeled(body_groups(node.frame()))[label]
        row, col = center(group)
        action = (6, col + offset[1], row + offset[0])
        node.step(*action)
        actions.append(action)
        print(
            "SEED", index, action, summary(node),
            "level", int(node.levels_completed), "terminal", node.terminal(),
        )

    candidates = [
        (6, col, row) for row in range(64) for col in range(64)
    ]
    outcomes = {}
    started = time.monotonic()
    for index, action in enumerate(candidates, 1):
        child = node.clone()
        child.step(*action)
        delay = index / 300 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        result = (
            int(child.levels_completed), child.terminal(), summary(child)
        )
        outcomes.setdefault(result, []).append(action)
        if int(child.levels_completed) > start_level:
            print("FOUND", action, result)
            print("ACTIONS", PREFIX + actions + [action])
            return
    print("ONE_STEP", len(candidates), len(outcomes))
    for result, equivalent in sorted(
        outcomes.items(), key=lambda item: (-item[0][0], -item[0][2][0])
    )[:16]:
        print("OUT", result, "via", equivalent[:4])
    print("ACTIONS", PREFIX + actions)


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
