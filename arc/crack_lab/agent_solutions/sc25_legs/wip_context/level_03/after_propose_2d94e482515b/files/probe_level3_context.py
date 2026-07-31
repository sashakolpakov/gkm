import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from legs import select_grid_cells_of_color
from players import play_level_1, play_level_2


def avatar(frame):
    board = P.arr(frame)
    mask = ((board == 9) | (board == 10))
    mask[34:] = False
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    values = board[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return (
        int(ys.min()), int(xs.min()),
        "/".join("".join(f"{int(v):X}" for v in row) for row in values),
    )


def fixed(frame):
    return tuple(
        (blob.color, blob.area, blob.bbox)
        for blob in P.connected_components(frame, colors=(4, 13), min_area=4)
        if blob.bbox[0] < 45
    )


def run(root, name, route, push):
    node = root.clone()
    for action in route:
        node.step(action)
    before = (avatar(node.frame()), fixed(node.frame()))
    select_grid_cells_of_color(
        node, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    accepted = (avatar(node.frame()), fixed(node.frame()))
    node.step(push)
    after = (avatar(node.frame()), fixed(node.frame()))
    print(name, "before", before, "accepted", accepted, "after", after,
          "level", node.levels_completed)


def probe(env):
    play_level_1(env)
    play_level_2(env)
    root = env.clone()
    contexts = {
        "gate": ([2, 2, 3, 3, 2], 2),
        "right_obstacle": ([4, 4, 4], 4),
        "left_wall": ([3, 3, 3], 3),
        "top_wall": ([1, 1, 1, 1], 1),
    }
    for name, (route, push) in contexts.items():
        run(root, name, route, push)

    node = root.clone()
    for action in (4, 4, 4):
        node.step(action)
    select_grid_cells_of_color(
        node, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    route = (3, 3, 3, 3, 2, 2, 2, 2, 3, 3)
    history = []
    for index, action in enumerate(route, 1):
        node.step(action)
        if node.levels_completed > root.levels_completed:
            history.append((index, action, "advanced", node.levels_completed))
            break
        history.append((index, action, avatar(node.frame()), node.levels_completed))
    print("post_gate", history)


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
