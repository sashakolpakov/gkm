"""Compare a leg's first reward action with all actions it emits."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_compact_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_repeated_frontier_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)
from perception import safe_step


SOLVERS = {
    4: solve_compact_bridge_carrier_peg_solitaire,
    6: solve_wrapped_bridge_carrier_peg_solitaire,
    7: solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    8: solve_grid_wrapped_bridge_carrier_peg_solitaire,
    9: solve_repeated_frontier_bridge_carrier_peg_solitaire,
}


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.base_level = int(inner.levels_completed)
        self.path = []
        self.first_completion = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        result = self.inner.step(action, *coordinates)
        if (self.first_completion is None
                and int(self.inner.levels_completed) > self.base_level):
            self.first_completion = len(self.path)
        return result


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "9"))
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
    SOLVERS[desired](recorder)
    print("completion", desired, recorder.first_completion,
          len(recorder.path), int(recorder.levels_completed), flush=True)


arena.run_program("lf52", probe)
