import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from legs import select_grid_cells_of_color
from players import play_level_1, play_level_2


def avatar(frame):
    a = P.arr(frame)
    mask = ((a == 9) | (a == 10))
    mask[34:] = False
    ys, xs = np.where(mask)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def fixed_pieces(frame):
    return tuple(
        (b.color, b.area, b.bbox)
        for b in P.connected_components(frame, colors=(4, 6, 13), min_area=4)
        if b.bbox[0] < 45
    )


def run(root, name, actions):
    node = root.clone()
    before_fixed = fixed_pieces(node.frame())
    events = []
    for index, action in enumerate(actions, 1):
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
        current_fixed = fixed_pieces(node.frame())
        if current_fixed != before_fixed or node.levels_completed != root.levels_completed:
            events.append((index, action, avatar(node.frame()), current_fixed,
                           node.levels_completed))
        before_fixed = current_fixed
    print(name, "avatar", avatar(node.frame()), "fixed", fixed_pieces(node.frame()),
          "level", node.levels_completed, "events", events)


def probe(env):
    play_level_1(env)
    play_level_2(env)
    roots = {"raw": env.clone()}
    accepted = env.clone()
    select_grid_cells_of_color(
        accepted, xs=(25, 30, 35), ys=(50, 55, 60), color=0
    )
    roots["accepted"] = accepted

    around_above = [1, 1, 4, 4, 4, 4, 2, 2]
    around_below = [2, 2, 4, 4, 4, 4, 1, 1]
    interactions = {
        "none": [],
        "right": [4],
        "use": [6],
        "click_device": [(6, 56, 24)],
        "right_use": [4, 6],
        "use_right": [6, 4],
    }
    back_and_exit = [2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2]

    for root_name, root in roots.items():
        run(root, f"{root_name}:above", around_above)
        run(root, f"{root_name}:below", around_below)
        for interaction_name, interaction in interactions.items():
            run(
                root,
                f"{root_name}:{interaction_name}:exit",
                around_above + interaction + back_and_exit,
            )


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)
