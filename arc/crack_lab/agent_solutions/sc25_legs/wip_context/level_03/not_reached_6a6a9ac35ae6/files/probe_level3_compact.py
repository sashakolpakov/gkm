import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from legs import select_grid_cells_of_color
from players import play_level_1, play_level_2


XS = (25, 30, 35)
YS = (50, 55, 60)


def grid(frame):
    return tuple(tuple(int(frame[y][x]) for x in XS) for y in YS)


def pieces(frame):
    return tuple(
        (b.color, b.area, b.bbox)
        for b in P.connected_components(frame, colors=(4, 6, 9, 10, 13), min_area=4)
    )


def spatial_delta(before, after):
    a, b = P.arr(before), P.arr(after)
    changed = a[:, :62] != b[:, :62]
    ys, xs = np.where(changed)
    if not len(ys):
        return (0, None)
    return (int(len(ys)), (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))


def step_any(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def summarize(root, name, actions):
    node = root.clone()
    before = node.frame()
    for action in actions:
        step_any(node, action)
    print(
        name,
        "n", len(actions),
        "level", node.levels_completed,
        "grid", grid(node.frame()),
        "delta", spatial_delta(before, node.frame()),
        "pieces", pieces(node.frame()),
    )
    return node


def probe(env):
    play_level_1(env)
    play_level_2(env)
    root = env.clone()
    print("start", env.levels_completed, grid(env.frame()), pieces(env.frame()))

    for action in (1, 2, 3, 4, 6):
        node = root.clone()
        before = node.frame()
        try:
            node.step(action)
            print("action", action, "delta", spatial_delta(before, node.frame()),
                  "pieces", pieces(node.frame()))
        except Exception as exc:
            print("action", action, "error", type(exc).__name__, str(exc))

    for point in ((25, 50), (30, 50)):
        node = root.clone()
        states = [grid(node.frame())]
        piece_states = [pieces(node.frame())]
        for _ in range(5):
            node.step(6, *point)
            states.append(grid(node.frame()))
            piece_states.append(pieces(node.frame()))
        print("cycle", point, states)
        print("cycle_pieces", point, piece_states)

    configs = {}
    for color in (0, 2):
        node = root.clone()
        select_grid_cells_of_color(node, XS, YS, color)
        configs[f"select_{color}"] = node
        print("config", color, grid(node.frame()), pieces(node.frame()))

    configs["raw"] = root
    routes = {
        "device": [4, 4, 4],
        "device_use": [4, 4, 6],
        "device_click": [4, 4, (6, 56, 24)],
        "gate": [2, 2, 3, 3, 2],
        "gate_use": [2, 2, 3, 3, 6, 2],
        "gate_click": [2, 2, 3, 3, (6, 29, 35), 2],
    }
    for config_name, config in configs.items():
        for route_name, actions in routes.items():
            summarize(config, f"{config_name}:{route_name}", actions)


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
