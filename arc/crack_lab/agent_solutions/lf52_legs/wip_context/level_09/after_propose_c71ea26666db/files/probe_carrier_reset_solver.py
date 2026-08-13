"""Solve a carrier peg board while allowing the discovered reset action 7."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_peg_solitaire_with_carrier
from perception import safe_step


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.inner.step(action, *coordinates)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "2"))
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = env.clone() if desired == 1 else None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current
    recorder = Recorder(entry)
    solve_peg_solitaire_with_carrier(
        recorder,
        max_states=int(os.environ.get("OPT_STATES", "5000")),
        max_depth=int(os.environ.get("OPT_DEPTH", "40")),
        include_reset=os.environ.get("OPT_RESET", "1") == "1",
        debug_progress=True,
    )
    print("reset_solution", desired, len(recorder.path),
          int(recorder.levels_completed), tuple(recorder.path), flush=True)


arena.run_program("lf52", probe)
