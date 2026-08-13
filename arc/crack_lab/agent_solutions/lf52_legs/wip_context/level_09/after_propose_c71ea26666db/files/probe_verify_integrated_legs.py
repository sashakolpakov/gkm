"""Verify action counts and rewards for the integrated compact legs."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_compact_bridge_carrier_peg_solitaire,
    solve_repeated_frontier_bridge_carrier_peg_solitaire,
)
from perception import safe_step


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    @property
    def levels_completed(self):
        return self.inner.levels_completed

    def frame(self):
        return self.inner.frame()

    def clone(self):
        return self.inner.clone()

    def step(self, *args):
        self.path.append(args[0] if len(args) == 1 else tuple(args))
        return self.inner.step(*args)


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    checked_l4 = False
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if not checked_l4 and prior < 3 <= current:
            recorder = Recorder(env.clone())
            solve_compact_bridge_carrier_peg_solitaire(recorder)
            print("level4", len(recorder.path),
                  int(recorder.levels_completed), flush=True)
            checked_l4 = True
        prior = current
    recorder = Recorder(env.clone())
    solve_repeated_frontier_bridge_carrier_peg_solitaire(recorder)
    print("level9", len(recorder.path), int(recorder.levels_completed), flush=True)


arena.run_program("lf52", probe)
