"""Reward-check existing fixed legs on earlier and frontier configurations."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_direct_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_multi_bridge_wrapped_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)


ENTRIES = {5: 149, 6: 238, 7: 331, 8: 476, 9: 544}
TRIALS = {
    5: (
        solve_direct_bridge_carrier_peg_solitaire,
        solve_wrapped_bridge_carrier_peg_solitaire,
        solve_grid_wrapped_bridge_carrier_peg_solitaire,
    ),
    6: (
        solve_direct_bridge_carrier_peg_solitaire,
        solve_grid_wrapped_bridge_carrier_peg_solitaire,
        solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    ),
    7: (
        solve_wrapped_bridge_carrier_peg_solitaire,
        solve_grid_wrapped_bridge_carrier_peg_solitaire,
        solve_multi_bridge_wrapped_carrier_peg_solitaire,
    ),
    8: (
        solve_wrapped_bridge_carrier_peg_solitaire,
        solve_parallel_wrapped_bridge_carrier_peg_solitaire,
        solve_multi_bridge_wrapped_carrier_peg_solitaire,
    ),
    9: (
        solve_wrapped_bridge_carrier_peg_solitaire,
        solve_grid_wrapped_bridge_carrier_peg_solitaire,
        solve_parallel_wrapped_bridge_carrier_peg_solitaire,
        solve_multi_bridge_wrapped_carrier_peg_solitaire,
    ),
}


class CountingEnv:
    def __init__(self, base):
        self.base = base
        self.steps = 0

    def step(self, *action):
        self.steps += 1
        return self.base.step(*action)

    def clone(self):
        return CountingEnv(self.base.clone())

    def __getattr__(self, name):
        return getattr(self.base, name)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    node = env.clone()
    entries = {}
    for index, action in enumerate(campaign):
        if index in ENTRIES.values():
            level = next(level for level, point in ENTRIES.items() if point == index)
            entries[level] = node.clone()
        play(node, action)
    entries[9] = node.clone()
    for level, solvers in TRIALS.items():
        for solver in solvers:
            trial = CountingEnv(entries[level].clone())
            before = np.asarray(trial.frame()).copy()
            solver(trial)
            print("TRIAL", {
                "level": level,
                "solver": solver.__name__,
                "steps": trial.steps,
                "reward": trial.levels_completed > level - 1,
                "result_level": trial.levels_completed,
                "changed": not np.array_equal(before, trial.frame()),
            }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
