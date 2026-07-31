"""Bounded contact experiments on pristine wa30 level 1 clones."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, connected_components, frame_delta


def cells_for(frame, color, min_area=1):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=[color], min_area=min_area)
    ]


def state(env):
    frame = env.frame()
    orange = cells_for(frame, 14)
    black = cells_for(frame, 0)
    blocks = cells_for(frame, 4, min_area=4)
    strip = cells_for(frame, 2, min_area=4)
    return {
        "level": int(env.levels_completed),
        "orange": orange,
        "black": black,
        "blocks": blocks,
        "strip": strip,
    }


TESTS = {
    "push_C_up": [1, 1, 1],
    "use_below_C": [1, 1, 5],
    "push_C_right": [3, 1, 1, 1, 4],
    "use_left_C": [3, 1, 1, 1, 5],
    "push_C_left": [4, 1, 1, 1, 3],
    "use_right_C": [4, 1, 1, 1, 5],
    "enter_strip_below": [1, 1, 3, 1, 1, 1],
    "use_below_strip": [1, 1, 3, 1, 1, 5],
}


def probe(env):
    for name, path in TESTS.items():
        clone = env.clone()
        trace = []
        previous = np.asarray(clone.frame()).copy()
        for action in path:
            clone.step(action)
            current = np.asarray(clone.frame())
            trace.append(
                (
                    ACTION_NAME[action],
                    frame_delta(previous, current)["count"],
                    state(clone),
                )
            )
            previous = current.copy()
        print(name, trace)


arena.run_program("wa30", probe)
