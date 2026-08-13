"""Tune the existing bridge/carrier leg for a shorter verified prefix path."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_bridge_carrier_peg_solitaire
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
        action = args[0] if len(args) == 1 else tuple(args)
        self.path.append(action)
        return self.inner.step(*args)


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def probe(env):
    desired_level = int(os.environ.get("OPT_LEVEL", "5"))
    with open("checkpoint.json") as stream:
        path = tuple(normalize(action) for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    for action in path:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired_level - 1 <= current:
            entry = env.clone()
            break
        prior = current

    settings = (
        (80, 4), (120, 4), (250, 4), (650, 4),
        (80, 8), (120, 8), (250, 8), (650, 8),
        (120, 12), (250, 12), (650, 12),
        (120, 24), (250, 24), (650, 24),
        (250, 48), (650, 48), (1000, 48),
    )
    results = []
    for max_states, lookahead in settings:
        recorder = Recorder(entry.clone())
        solve_bridge_carrier_peg_solitaire(
            recorder, max_align_states=max_states,
            alignment_lookahead=lookahead,
        )
        won = int(recorder.levels_completed) >= desired_level
        result = (len(recorder.path) if won else None,
                  max_states, lookahead, tuple(recorder.path) if won else ())
        results.append(result)
        print("setting", result[:3], flush=True)
    wins = [result for result in results if result[0] is not None]
    print("best", min(wins) if wins else None, flush=True)


arena.run_program("lf52", probe)
