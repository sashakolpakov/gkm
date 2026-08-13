"""Test whether an existing fixed leg cheaply stages the level-5 planner."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_bridge_carrier_peg_solitaire,
    solve_direct_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
)


ENTRY_INDEX = 149


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
    for action in campaign[:ENTRY_INDEX]:
        play(env, action)
    entry = env.clone()
    for staging in (
        solve_direct_bridge_carrier_peg_solitaire,
        solve_grid_wrapped_bridge_carrier_peg_solitaire,
    ):
        trial = CountingEnv(entry.clone())
        staging(trial)
        staged_steps = trial.steps
        staged_level = trial.levels_completed
        solve_bridge_carrier_peg_solitaire(trial, max_align_states=650)
        print("TRIAL", {
            "staging": staging.__name__,
            "staged_steps": staged_steps,
            "staged_level": staged_level,
            "total_steps": trial.steps,
            "result_level": trial.levels_completed,
            "reward": trial.levels_completed > 4,
        }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
