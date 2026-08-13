"""Record one parameterized run of the reusable bridge/carrier planner."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_bridge_carrier_peg_solitaire
from perception import safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


class Recorder:
    def __init__(self, env):
        self.env = env
        self.path = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clone(self):
        return self.env.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.env.step(action, *coordinates)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    for action in campaign:
        if int(env.levels_completed) >= desired - 1:
            break
        safe_step(env, action)
    recorder = Recorder(env.clone())
    solve_bridge_carrier_peg_solitaire(
        recorder,
        max_align_states=int(os.environ.get("OPT_STATES", "120")),
        max_macros=int(os.environ.get("OPT_MACROS", "40")),
        alignment_lookahead=int(os.environ.get("OPT_LOOKAHEAD", "24")),
        reverse_capture_ties=os.environ.get("OPT_REVERSE_CAPTURES") == "1",
        reverse_bridge_ties=os.environ.get("OPT_REVERSE_BRIDGES") == "1",
    )
    print("variant", desired, os.environ.get("OPT_STATES", "120"),
          os.environ.get("OPT_LOOKAHEAD", "24"), len(recorder.path),
          int(recorder.levels_completed), tuple(recorder.path), flush=True)


arena.run_program("lf52", probe)
