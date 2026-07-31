"""Trace the verified level-5 movable-assembly interactions symbolically."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import players


def signature(env, color, min_area):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(color,), min_area=min_area
        )
        if blob.bbox[1] < 40 and blob.area < 500
    )


def avatar(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=1
        )
        if blob.bbox[1] < 40
    )


class Trace:
    def __init__(self, env):
        self.env = env
        self.index = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *action):
        self.index += 1
        before_level = self.env.levels_completed
        before8 = signature(self.env, 8, 4)
        before0 = signature(self.env, 0, 4)
        before_avatar = avatar(self.env)
        result = self.env.step(*action)
        after8 = signature(self.env, 8, 4)
        after0 = signature(self.env, 0, 4)
        after_avatar = avatar(self.env)
        if (
            before8 != after8
            or before0 != after0
            or any(area < 4 for _, area in after_avatar)
            or self.env.levels_completed > before_level
        ):
            print(
                "L5_TRACE", self.index, action,
                "A", before_avatar, "TO", after_avatar,
                "C8", before8, "TO", after8,
                "C0", before0, "TO", after0,
                "LEVEL", self.env.levels_completed,
            )
        return result


def observe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    players.play_level_5(Trace(env))


arena.run_program("dc22", observe)
