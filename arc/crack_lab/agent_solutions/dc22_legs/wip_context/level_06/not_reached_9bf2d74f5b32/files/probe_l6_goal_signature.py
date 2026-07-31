"""Observe the pre-reward state and final action of the solved dc22 levels."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


class TransitionObserver:
    def __init__(self, env):
        self.env = env
        self.level = env.levels_completed
        self.signature = self.goal_tiles(env.frame())

    @staticmethod
    def goal_tiles(frame):
        return tuple(
            blob.bbox
            for blob in perception.connected_components(
                frame, colors=(11,), min_area=4
            )
            if blob.area == 4
            and blob.size == (2, 2)
            and blob.bbox[1] < 40
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *action):
        level = self.env.levels_completed
        before = perception.arr(self.env.frame()).copy()
        blobs = [
            (blob.color, blob.bbox, blob.area)
            for blob in perception.connected_components(before, min_area=1)
            if blob.color == 14
        ]
        result = self.env.step(*action)
        after_signature = self.goal_tiles(self.env.frame())
        if self.env.levels_completed == level:
            if after_signature != self.signature:
                print(
                    "EXIT_SIGNATURE_CHANGE", level + 1,
                    "ACTION", action,
                    "AVATAR_BEFORE", blobs,
                    "FROM", self.signature,
                    "TO", after_signature,
                )
            self.signature = after_signature
        if self.env.levels_completed > level:
            print(
                "LEVEL_TRANSITION", level + 1,
                "ACTION", action,
                "AVATAR_BEFORE", blobs,
            )
            for _, (r0, c0, r1, c1), _ in blobs:
                lo_r, hi_r = max(0, r0 - 3), min(62, r1 + 3)
                lo_c, hi_c = max(0, c0 - 3), min(63, c1 + 3)
                print(
                    "GOAL_NEIGHBORHOOD",
                    (lo_r, lo_c, hi_r, hi_c),
                    before[lo_r:hi_r + 1, lo_c:hi_c + 1].tolist(),
                )
            self.level = self.env.levels_completed
            self.signature = after_signature
        return result


def observe(env):
    solve.solve(TransitionObserver(env))


arena.run_program("dc22", observe)
