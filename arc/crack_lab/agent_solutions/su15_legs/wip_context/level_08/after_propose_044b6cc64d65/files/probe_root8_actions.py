import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_finish8 import PREFIX, centers


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start = env.levels_completed
    ring_mask = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(env.frame()[row][col]) == 9
    }
    for action in PREFIX:
        env.step(*action)
    frame = env.frame()
    actions = {
        (6, col, row)
        for row in range(10, 64)
        for col in range(64)
        if int(frame[row][col]) == 7
    }
    actions |= {
        (6, col, row)
        for row, col in ((19, 56), (55, 7), (55, 56), (32, 32))
    }
    outcomes = {}
    for action in sorted(actions):
        clone = env.clone()
        clone.step(*action)
        square = tuple(
            blob.bbox
            for blob in connected_components(
                clone.frame(), colors=(8,), min_area=9
            )
            if blob.bbox[0] >= 10
        )
        result = (
            clone.levels_completed, clone.terminal(),
            square, centers(clone.frame(), 7),
            sum(
                (row, col) in ring_mask
                for row in range(10, 64)
                for col in range(64)
                if int(clone.frame()[row][col]) in (7, 8)
            ),
        )
        outcomes.setdefault(result, []).append(action)
    print("root", centers(frame, 7), "unique", len(outcomes))
    for result, result_actions in sorted(
        outcomes.items(), key=lambda item: -item[0][-1]
    ):
        print(result, "actions", result_actions)


A.run_program("su15", inspect)
