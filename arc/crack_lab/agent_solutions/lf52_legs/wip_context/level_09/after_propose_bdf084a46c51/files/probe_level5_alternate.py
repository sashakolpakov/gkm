"""Verify the early-capture level-5 branch and solve its suffix."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    _bridge_carrier_moves, _bridge_carrier_state,
    solve_bridge_carrier_peg_solitaire,
)
from perception import safe_step


STAGE2_PREFIX = 166
EARLY_KEYS = (2, 2, 3, 2, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 2)
EARLY_MOVE = ((24, 30), (24, 42))


class CountedEnv:
    def __init__(self, env):
        self.env = env
        self.moves = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clone(self):
        return self.env.clone()

    def step(self, *action):
        self.moves += 1
        return self.env.step(*action)


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:STAGE2_PREFIX]:
        env.step(action)

    counted = CountedEnv(env.clone())
    for action in EARLY_KEYS:
        safe_step(counted, action)
    print("BEFORE_CAPTURE", {"moves": _bridge_carrier_moves(counted.frame()),
                             "pegs": tuple(sorted(
                                 _bridge_carrier_state(counted.frame())[1]
                             ))})
    for position in EARLY_MOVE:
        safe_step(counted, (6, position[1] + 1, position[0] + 1))
    state = _bridge_carrier_state(counted.frame())
    print("AFTER_CAPTURE", {"moves": counted.moves,
                            "pegs": tuple(sorted(state[1])),
                            "level": counted.levels_completed})

    if os.environ.get("CHECK_ONLY") == "1":
        return

    solve_bridge_carrier_peg_solitaire(counted, max_align_states=650)
    print("ALTERNATE_RESULT", {"suffix_actions": counted.moves,
                               "level": counted.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
