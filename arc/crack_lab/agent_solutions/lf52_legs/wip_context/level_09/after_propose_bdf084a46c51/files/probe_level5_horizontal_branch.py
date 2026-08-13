"""Verify the shorter horizontal branch after level-5 macro eight."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import solve_bridge_carrier_peg_solitaire
from perception import safe_step


PREFIX = 208


class CountedEnv:
    def __init__(self, env):
        self.env = env
        self.moves = 0
        self.path = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clone(self):
        return self.env.clone()

    def step(self, *action):
        self.moves += 1
        self.path.append(action[0] if len(action) == 1 else tuple(action))
        return self.env.step(*action)


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:PREFIX]:
        safe_step(env, action)
    counted = CountedEnv(env.clone())
    for action in (1, (6, 46, 19), (6, 34, 19)):
        safe_step(counted, action)
    trace = []
    solve_bridge_carrier_peg_solitaire(
        counted, max_align_states=650, trace=trace,
    )
    print("HORIZONTAL_RESULT", {"level": counted.levels_completed,
                                "suffix_actions": counted.moves,
                                "path": counted.path, "trace": trace})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
