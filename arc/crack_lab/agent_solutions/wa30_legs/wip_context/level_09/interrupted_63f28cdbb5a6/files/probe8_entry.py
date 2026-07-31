"""Compact pristine level-8 observation for joint cap optimization."""

import gkm_try

from perception import color_counts, connected_components
from probe9_verify import boxes, tile_map


class ReachedLevel8(Exception):
    pass


class StopAtLevel8:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 7:
            raise ReachedLevel8
        return result


def compact(env):
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "actions": tuple(env.actions),
        "colors": color_counts(env.frame()),
        "actors": {
            color: boxes(env.frame(), color)
            for color in (4, 12, 14, 15)
        },
        "blobs": tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(env.frame(), min_area=4)
            if blob.color not in (1, 2)
        ),
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    print("L8_ENTRY", compact(env), flush=True)
    print("L8_MAP", *tile_map(env.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
