"""Short contextual action probes from the pristine level-5 entry."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_level5 import reach_level_5


def state(env):
    frame = np.asarray(env.frame())
    nines = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=1)
        if blob.bbox[0] < 55 and blob.bbox[1] >= 4
    ]
    large_nine = max(nines, key=lambda blob: (blob.area == 15, blob.area))
    zeros = [
        blob.bbox
        for blob in connected_components(frame, colors=(0,), min_area=1)
        if blob.bbox[0] < 60 and blob.bbox[1] >= 4
    ]
    refills = [
        blob.bbox
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[0] < 60
    ]
    portrait = frame[53:63, 1:11]
    hud = max(
        (8, 9, 12, 14),
        key=lambda color: int(np.count_nonzero(portrait == color)),
    )
    return (
        int(env.levels_completed),
        large_nine.bbox,
        tuple(zeros),
        tuple(refills),
        int(np.count_nonzero(frame[60:, :] == 11)),
        hud,
    )


def trace(root, path):
    clone = root.clone()
    states = [state(clone)]
    for action in path:
        if clone.terminal():
            break
        clone.step(action)
        states.append(state(clone))
    return states


def inspect(env):
    reach_level_5(env)
    root = env.clone()
    probes = (
        (1, 1, 1, 1, 1),
        (2, 2, 2),
        (3, 3, 3, 3, 3),
        (4, 4, 4, 4, 4),
    )
    for path in probes:
        print("trace", "".join(map(str, path)), trace(root, path))
    for first in env.actions:
        for second in env.actions:
            path = (first, second)
            print("pair", "".join(map(str, path)), trace(root, path)[-1])
    central_refill_target = (
        1, 3, 1, 1, 3, 3, 3,
        4, 3, 4, 3,
        4, 4, 4,
        3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 1,
    )
    checkpoint_switch_target = (
        1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3, 3, 3,
        4, 4, 4, 2, 4, 4, 4, 2, 4, 2, 3, 3, 3,
        4, 3, 4, 3,
        4, 4, 4,
        3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 1,
    )
    directional_checkpoint_target = (
        1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1,
        2, 2, 2, 2, 2, 2, 4, 2,
        1, 4, 4, 4, 4, 1, 4, 4, 1, 1, 3, 3, 3,
        4, 3, 4, 3,
        4, 4, 4,
        3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 1,
    )
    shape_color_target = (
        1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3,
        3, 3, 4, 4,
        2, 2, 2, 2, 2, 2, 4, 2,
        1, 1, 4, 3,
        2, 4, 4, 4, 4, 1, 4, 4, 1, 1, 3, 4,
        3, 3, 3, 4, 3, 4, 3,
        4, 1, 2, 2, 2, 4, 2, 3, 2, 2, 1,
    )
    for path in (
        (3, 2, 2),
        (3, 2, 2, 1),
        (3, 2, 2, 1, 1),
        central_refill_target,
        checkpoint_switch_target,
        directional_checkpoint_target,
        shape_color_target,
    ):
        clone = root.clone()
        for action in path:
            if clone.terminal():
                break
            clone.step(action)
        print(
            "candidate",
            "".join(map(str, path)),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "state",
            state(clone),
        )


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
