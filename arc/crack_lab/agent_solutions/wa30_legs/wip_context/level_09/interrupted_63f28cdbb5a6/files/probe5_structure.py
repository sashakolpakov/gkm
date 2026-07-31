"""Inspect level 5 at the reusable leg's phase boundaries."""

import gkm_try

from probe_minimize_segments import CaptureSegments
from probe9_verify import boxes, tile_map


PHASES = (
    ("direct", [4] + [2] * 5 + [3, 5] + [1] * 6 + [3] * 8 + [5]),
    (
        "upper_left",
        [4] * 9 + [1] * 7 + [3, 5] + [2] * 6 + [3, 5, 2],
    ),
    (
        "upper_right",
        [4] * 3 + [1] * 6 + [3, 5] + [2] * 5
        + [3] * 3 + [5, 2],
    ),
    (
        "boundary",
        [4] * 2 + [2] * 5 + [4, 5] + [1] * 5
        + [3] * 10 + [2, 5, 1],
    ),
)


def summary(env, turn):
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "courier": boxes(env.frame(), 12),
    }


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    clone = capture.starts[4].clone()
    print("L5_PHASE", "start", summary(clone, 0), flush=True)
    print(*tile_map(clone.frame()), sep="\n", flush=True)
    turn = 0
    for label, actions in PHASES:
        for action in actions:
            clone.step(action)
            turn += 1
        print("L5_PHASE", label, summary(clone, turn), flush=True)
        print(*tile_map(clone.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
