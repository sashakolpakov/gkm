"""Reproduce the solved level-4 mechanism through compact observations."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components
from probe_level5 import tile_map


def reach_level_4(env):
    while env.levels_completed < 3 and not env.terminal():
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)


def body_box(frame):
    candidates = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=3)
        if blob.bbox[0] < 55 and blob.bbox[1] >= 4
    ]
    return max(candidates, key=lambda blob: (blob.area == 15, blob.area)).bbox


def compact(env):
    frame = np.asarray(env.frame())
    portrait = frame[53:63, 1:11]
    hud = max(
        (8, 9, 12, 14),
        key=lambda color: int(np.count_nonzero(portrait == color)),
    )
    refills = len(
        [
            blob
            for blob in connected_components(frame, colors=(11,), min_area=4)
            if blob.bbox[0] < 60
        ]
    )
    return (
        body_box(frame),
        hud,
        int(np.count_nonzero(frame[60:, :] == 11)) // 4,
        refills,
        int(env.levels_completed),
    )


class LoggingEnv:
    def __init__(self, env):
        self.env = env
        self.steps = 0

    def terminal(self):
        return self.env.terminal()

    def step(self, action):
        before = compact(self.env)
        self.env.step(action)
        self.steps += 1
        after = compact(self.env)
        print(self.steps, action, before, "->", after)


def inspect(env):
    reach_level_4(env)
    print("entry", compact(env))
    rows, signatures = tile_map(env.frame())
    print("map")
    print("\n".join(rows))
    print(
        "specials",
        [
            (cell, signature)
            for cell, signature in signatures.items()
            if any(color in {0, 8, 9, 11, 12, 14} for color, _ in signature)
        ],
    )
    players.play_level_4(LoggingEnv(env.clone()))


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
