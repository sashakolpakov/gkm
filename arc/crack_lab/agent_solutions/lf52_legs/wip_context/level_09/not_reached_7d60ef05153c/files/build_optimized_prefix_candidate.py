"""Build a validated, non-checkpoint prefix candidate from verified shorter legs."""

import json

import gkm_try
from legs import (
    solve_bridge_carrier_peg_solitaire,
    solve_long_coherent_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)


OUTPUT = "optimized_prefix_l4_l6_candidate.json"
FULL_OUTPUT = "optimized_campaign_l4_l6_l9_candidate.json"


class Recorder:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def clone(self):
        return Recorder(self.env.clone())

    def step(self, *action):
        public = action[0] if len(action) == 1 else tuple(action)
        self.actions.append(public)
        return self.env.step(*action)

    def __getattr__(self, name):
        return getattr(self.env, name)


def build(env):
    with open("checkpoint.json") as checkpoint_file:
        original = json.load(checkpoint_file)["final_path"]
    root = env.clone()

    level_four_entry = root.clone()
    for action in original[:87]: level_four_entry.step(action)
    level_four = Recorder(level_four_entry)
    solve_bridge_carrier_peg_solitaire(level_four, reverse_choices=True)

    level_six_entry = root.clone()
    for action in original[:238]: level_six_entry.step(action)
    level_six = Recorder(level_six_entry)
    solve_wrapped_bridge_carrier_peg_solitaire(level_six)

    candidate = (
        original[:87] + level_four.actions + original[149:238]
        + level_six.actions + original[331:544]
    )
    valid = gkm_try.A.validate("lf52", candidate, 8)
    payload = {
        "game": "lf52",
        "validated": bool(valid),
        "levels": 8,
        "moves": len(candidate),
        "replacements": {"level_4": len(level_four.actions), "level_6": len(level_six.actions)},
        "final_path": candidate,
    }
    with open(OUTPUT, "w") as output_file:
        json.dump(payload, output_file, separators=(",", ":"))
    print("PREFIX_CANDIDATE", OUTPUT, len(candidate), valid)

    level_nine_entry = root.clone()
    for action in candidate:
        if isinstance(action, tuple): level_nine_entry.step(*action)
        else: level_nine_entry.step(action)
    level_nine = Recorder(level_nine_entry)
    solve_long_coherent_bridge_carrier_peg_solitaire(level_nine)
    full_candidate = candidate + level_nine.actions
    full_valid = gkm_try.A.validate("lf52", full_candidate, 9)
    full_payload = {
        "game": "lf52",
        "validated": bool(full_valid),
        "levels": 9,
        "moves": len(full_candidate),
        "prefix_moves": len(candidate),
        "level_9_moves": len(level_nine.actions),
        "final_path": full_candidate,
    }
    with open(FULL_OUTPUT, "w") as output_file:
        json.dump(full_payload, output_file, separators=(",", ":"))
    print("FULL_CANDIDATE", FULL_OUTPUT, len(full_candidate), full_valid)


gkm_try.A.run_program("lf52", build)
