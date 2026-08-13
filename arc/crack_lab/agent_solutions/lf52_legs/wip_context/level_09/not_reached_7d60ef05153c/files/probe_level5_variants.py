"""Compare verified bridge/carrier solver paths from pristine level 5."""

import json
import os

import gkm_try
from legs import solve_bridge_carrier_peg_solitaire


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


class Recorder:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def clone(self):
        return Recorder(self.env.clone())

    def step(self, *action):
        recorded = action[0] if len(action) == 1 else tuple(action)
        self.actions.append(recorded)
        return self.env.step(*action)

    def __getattr__(self, name):
        return getattr(self.env, name)


def compare(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"][:LEVEL_ENDS[TARGET_LEVEL - 1]]
    for action in prefix:
        env.step(action)
    entry = env.clone()

    results = []
    lookaheads = tuple(int(value) for value in os.environ.get("LOOKAHEADS", "0").split(","))
    reverse_choices = os.environ.get("REVERSE_CHOICES") == "1"
    max_align_states = int(os.environ.get("MAX_ALIGN_STATES", "650"))
    prefer_farthest_bridge = os.environ.get("FARTHEST_BRIDGE") == "1"
    reverse_captures = (
        None if "REVERSE_CAPTURES" not in os.environ
        else os.environ.get("REVERSE_CAPTURES") == "1"
    )
    reverse_bridge_ties = (
        None if "REVERSE_BRIDGE_TIES" not in os.environ
        else os.environ.get("REVERSE_BRIDGE_TIES") == "1"
    )
    for lookahead in lookaheads:
        recorder = Recorder(entry.clone())
        solve_bridge_carrier_peg_solitaire(
            recorder,
            max_align_states=max_align_states,
            alignment_lookahead=lookahead,
            reverse_choices=reverse_choices,
            prefer_farthest_bridge=prefer_farthest_bridge,
            reverse_captures=reverse_captures,
            reverse_bridge_ties=reverse_bridge_ties,
        )
        results.append((
            lookahead,
            recorder.levels_completed,
            len(recorder.actions),
            recorder.actions,
        ))
    print("L5_VARIANTS", [(a, level, length) for a, level, length, _ in results])
    winners = [result for result in results if result[1] >= TARGET_LEVEL]
    if winners:
        best = min(winners, key=lambda result: result[2])
        print("L5_BEST", best)


gkm_try.A.run_program("lf52", compare)
