"""Reproduce the pristine level-9 frame through the documented arena surface."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from perception import action_deltas, color_counts, connected_components


class ReachedLevel9(Exception):
    pass


class StopAtLevel9:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 8:
            raise ReachedLevel9
        return result


def components(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=4)
        if blob.area < 3000
    )


def inspect(env):
    try:
        solve.solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    print(
        "ENTRY",
        {
            "level": env.levels_completed,
            "actions": tuple(env.actions),
            "colors": color_counts(env.frame()),
            "components": components(env.frame()),
        },
        flush=True,
    )
    deltas = action_deltas(env, env.actions)
    for action in env.actions:
        child = env.clone()
        child.step(action)
        print(
            "ACTION",
            action,
            {
                "delta": (
                    deltas[action]["count"],
                    deltas[action]["bbox"],
                ),
                "components": components(child.frame()),
            },
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("wa30", inspect)
