"""Inspect and shorten the last courier leg of the 114-turn level-8 route."""

import gkm_try

from legs import disable_competing_couriers_and_expedite_paired_depots
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_stage import compact
from probe9_verify import tile_map


class ReachedTurn(Exception):
    pass


class StopAtTurn:
    def __init__(self, env, limit):
        self.env = env
        self.limit = limit
        self.turn = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        self.turn += 1
        if self.turn >= self.limit:
            raise ReachedTurn
        return result


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    clone = env.clone()
    stopped = StopAtTurn(clone, 99)
    try:
        disable_competing_couriers_and_expedite_paired_depots(stopped)
    except ReachedTurn:
        pass
    print("OPT8_TAIL", compact(clone, stopped.turn), flush=True)
    print(*tile_map(clone.frame()), sep="\n", flush=True)
    base_level = clone.levels_completed
    for turn in range(100, 115):
        clone.step(5)
        print(
            "OPT8_WAIT",
            turn,
            compact(clone, turn),
            flush=True,
        )
        if clone.levels_completed > base_level:
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
