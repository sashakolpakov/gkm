"""Trace level-8 target progress under the current reusable leg."""

import gkm_try

from legs import disable_competing_couriers_and_expedite_paired_depots
from perception import arr
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe9_verify import boxes


TARGET = {
    *((row, col) for row in (2, 3) for col in range(11, 15)),
    *((row, col) for row in range(12, 15) for col in range(12, 15)
      if (row, col) != (13, 13)),
}


def target_state(frame):
    grid = arr(frame)
    empty = []
    filled = []
    for row, col in sorted(TARGET):
        colors = set(
            int(value)
            for value in grid[
                row * 4:row * 4 + 4,
                col * 4:col * 4 + 4,
            ].flat
        )
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        if 4 in colors and 9 in colors:
            filled.append((row, col))
    return tuple(empty), tuple(filled)


class WonLevel8(Exception):
    pass


class TraceEnv:
    def __init__(self, env):
        self.env = env
        self.turn = 0
        self.prior = target_state(env.frame())

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.turn += 1
        current = target_state(self.env.frame())
        if current != self.prior:
            print(
                "L8_EVENT",
                self.turn,
                action,
                current,
                {
                    "avatar": boxes(self.env.frame(), 14),
                    "helpers": boxes(self.env.frame(), 12),
                    "competitors": boxes(self.env.frame(), 15),
                },
                flush=True,
            )
        self.prior = current
        if self.env.levels_completed > before:
            raise WonLevel8
        return result


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    traced = TraceEnv(env.clone())
    try:
        disable_competing_couriers_and_expedite_paired_depots(traced)
    except WonLevel8:
        pass
    print(
        "L8_ROUTE",
        traced.turn,
        traced.env.levels_completed,
        target_state(traced.env.frame()),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
