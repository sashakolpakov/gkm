import itertools
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


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
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def overlap(node):
        return len(
            {
                point
                for group in body_groups(node.frame())
                for point in group
            } & ring_mask
        )

    patterns = {
        "top_right_center": ((6, 56, 19),),
        "bottom_left_center": ((6, 7, 55),),
        "bottom_right_center": ((6, 56, 55),),
        "top_edge": ((6, 56, 15),),
        "all_centers": (
            (6, 56, 19), (6, 7, 55), (6, 56, 55),
        ),
    }
    for name, pattern in patterns.items():
        node = root.clone()
        best = (overlap(node), 0)
        stopped = False
        for index, action in enumerate(itertools.islice(
            itertools.cycle(pattern), 80
        ), 1):
            try:
                node.step(*action)
            except Exception:
                print(name, "stopped", index, "best", best)
                stopped = True
                break
            current = overlap(node)
            best = max(best, (current, -index))
            if int(node.levels_completed) > start_level:
                print("FOUND", name, index)
                return
        if not stopped:
            print(name, "best", (best[0], -best[1]), "final", overlap(node))


print("DONE", A.run_program("su15", inspect))
