"""Test shorter existing wrapped legs on another confirmed wrapped level."""

import json
import os

import gkm_try
from legs import (
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_long_coherent_bridge_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


class Recorder:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def clone(self):
        return Recorder(self.env.clone())

    def step(self, *action):
        self.actions.append(action[0] if len(action) == 1 else tuple(action))
        return self.env.step(*action)

    def __getattr__(self, name):
        return getattr(self.env, name)


def compare(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:LEVEL_ENDS[TARGET_LEVEL - 1]]:
        env.step(action)
    entry = env.clone()
    variants = (
        ("wrapped", solve_wrapped_bridge_carrier_peg_solitaire),
        ("parallel", solve_parallel_wrapped_bridge_carrier_peg_solitaire),
        ("grid", solve_grid_wrapped_bridge_carrier_peg_solitaire),
        ("long", solve_long_coherent_bridge_carrier_peg_solitaire),
    )
    results = []
    for name, solver in variants:
        recorder = Recorder(entry.clone())
        solver(recorder)
        results.append((name, recorder.levels_completed, len(recorder.actions)))
    print("CROSS_REUSE", TARGET_LEVEL, results)


gkm_try.A.run_program("lf52", compare)
