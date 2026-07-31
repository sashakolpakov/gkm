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
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def devices(frame):
    return tuple(
        (blob.color, blob.area, blob.bbox)
        for blob in P.connected_components(frame, colors=(4, 6, 13), min_area=4)
        if blob.bbox[0] < 45
    )


def execute(root, name, route):
    node = root.clone()
    last = (avatar(node.frame()), devices(node.frame()))
    events = []
    for index, action in enumerate(route, 1):
        node.step(*action) if isinstance(action, tuple) else node.step(action)
        now = (avatar(node.frame()), devices(node.frame()))
        if now != last or node.levels_completed != root.levels_completed:
            events.append((index, action, now, node.levels_completed))
        last = now
    print(name, "end", last, "level", node.levels_completed, "events", events)


def probe(env):
    play_level_1(env)
    play_level_2(env)
    raw = env.clone()
    accepted = env.clone()
    select_grid_cells_of_color(
        accepted, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    routes = {
        "left": [1, 1, 4, 4, 4, 4, 2, 2, 4],
        "top": [1, 1, 4, 4, 4, 4, 4, 2],
        "bottom": [2, 2, 4, 4, 4, 4, 4, 1],
        "right": [1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 3],
    }
    interactions = {
        "push": [],
        "use": [6],
        "click": [(6, 56, 24)],
    }
    for root_name, root in (("raw", raw), ("accepted", accepted)):
        for side, route in routes.items():
            for interaction, suffix in interactions.items():
                execute(root, f"{root_name}:{side}:{interaction}", route + suffix)


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
