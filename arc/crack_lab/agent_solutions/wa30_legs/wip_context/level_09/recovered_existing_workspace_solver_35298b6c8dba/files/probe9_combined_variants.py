"""Test seven-action thief/cargo collision routes from turn 40."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


def avatar_cell(frame):
    avatar = boxes(frame, 14)
    if not avatar:
        return None
    row0, col0, row1, col1 = avatar[0]
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def compact(env):
    grid = arr(env.frame())
    return {
        "avatar": avatar_cell(env.frame()),
        "avatar_box": boxes(env.frame(), 14),
        "thief": boxes(env.frame(), 15),
        "lower_cell": tuple(sorted(set(
            int(value) for value in grid[32:36, 4:8].flat
        ))),
        "target": target_state(env.frame()),
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    base = env.clone()
    for action in direct_second_prefix():
        base.step(action)
    candidates = {
        "direct_use": [2, 2] + [3] * 4 + [5],
        "dismiss_then_contact": [2, 2, 3, 5] + [3] * 3,
        "late_use": [2, 2] + [3] * 3 + [5, 3],
        "early_use": [2, 2, 5] + [3] * 4,
    }
    for name, path in candidates.items():
        clone = base.clone()
        for action in path:
            clone.step(action)
        print("COMBINED_VARIANT", name, path, compact(clone), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
