import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from legs import clear_level_5_platform_transfer
from perception import frame_delta


def avatar(frame):
    rows, cols = np.where(np.asarray(frame)[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def local_tiles(frame, center, radius=2):
    row, col = center
    out = {}
    for tile_row in range(row - radius, row + radius + 1):
        for tile_col in range(col - radius, col + radius + 1):
            block = np.asarray(frame)[
                2 * tile_row:2 * tile_row + 2,
                2 * tile_col:2 * tile_col + 2,
            ]
            out[(tile_row, tile_col)] = tuple(int(value) for value in block.ravel())
    return out


class RewardTrace:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *action):
        before_level = self.env.levels_completed
        before = np.asarray(self.env.frame()).copy()
        before_avatar = avatar(before)
        self.env.step(*action)
        if self.env.levels_completed > before_level:
            after = np.asarray(self.env.frame())
            print(
                "REWARD_TRANSITION",
                "action", action,
                "avatar", before_avatar, "to", avatar(after),
                "local_before", local_tiles(before, before_avatar, radius=1),
                "delta", {
                    key: value
                    for key, value in frame_delta(before, after).items()
                    if key != "samples"
                },
                "levels", self.env.levels_completed,
                flush=True,
            )


def run(env):
    for level in range(1, 6):
        print(
            f"LEVEL{level}_ENTRY", env.levels_completed,
            "avatar", avatar(env.frame()), flush=True,
        )
        getattr(players, f"play_level_{level}")(RewardTrace(env))


arena.run_program("dc22", run)
