import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from legs import select_grid_cells_of_color
from players import play_level_1, play_level_2


def pieces(frame):
    return tuple(
        (b.color, b.area, b.bbox)
        for b in P.connected_components(frame, colors=(4, 6, 9, 10, 13), min_area=4)
        if b.bbox[0] < 45
    )


def run(root, name, actions):
    node = root.clone()
    history = []
    old = pieces(node.frame())
    for index, action in enumerate(actions, 1):
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
        new = pieces(node.frame())
        if new != old or node.levels_completed != root.levels_completed:
            history.append((index, action, new, node.levels_completed))
        old = new
    print(name, "changes", history, "final", pieces(node.frame()))
    return node


def probe(env):
    play_level_1(env)
    play_level_2(env)
    roots = {"raw": env.clone()}
    accepted = env.clone()
    select_grid_cells_of_color(
        accepted, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    roots["accepted"] = accepted

    routes = {
        "push_upper": [1, 4, 4, 4, 4],
        "push_lower": [2, 4, 4, 4, 4],
        "around_contact": [1, 4, 4, 4, 4, 2, 4],
        "around_use": [1, 4, 4, 4, 4, 2, 6, 4],
        "around_click": [1, 4, 4, 4, 4, 2, (6, 56, 24), 4],
    }
    for root_name, root in roots.items():
        for route_name, route in routes.items():
            run(root, f"{root_name}:{route_name}", route)

    # If contacting the right device changes latent state, return to the barrier
    # and try the shortest crossing/exit route.
    outward = [1, 4, 4, 4, 4, 2]
    returns = {
        "touch": [4],
        "use": [6],
        "click": [(6, 56, 24)],
        "touch_use": [4, 6],
    }
    back_to_exit = [1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2]
    for root_name, root in roots.items():
        for interaction_name, interaction in returns.items():
            run(
                root,
                f"{root_name}:device_{interaction_name}_then_exit",
                outward + interaction + back_to_exit,
            )


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
