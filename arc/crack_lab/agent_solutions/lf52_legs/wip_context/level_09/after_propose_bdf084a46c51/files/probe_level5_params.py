"""Measure level-5 solver cost under one alignment-lookahead setting."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import solve_bridge_carrier_peg_solitaire


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
    target_level = int(os.environ.get("TARGET_LEVEL", "5"))
    prefix = 87 if target_level == 4 else 149
    for action in path[:prefix]:
        env.step(action)
    lookahead = int(os.environ.get("LOOKAHEAD", "0"))
    max_align = int(os.environ.get(
        "MAX_ALIGN", "120" if target_level == 4 else "650"
    ))
    counted = CountedEnv(env.clone())
    low_macro = int(os.environ.get("LOW_MACRO", "-1"))
    low_value = int(os.environ.get("LOW_VALUE", str(lookahead)))
    schedule = None
    if low_macro >= 0:
        schedule = [lookahead] * 40
        schedule[low_macro] = low_value
    high_macros = tuple(
        int(value) for value in os.environ.get("HIGH_MACROS", "").split(",")
        if value
    )
    if high_macros:
        schedule = [lookahead] * 40
        for macro_index in high_macros:
            schedule[macro_index] = low_value
    trace = []
    solve_bridge_carrier_peg_solitaire(
        counted, max_align_states=max_align,
        alignment_lookahead=lookahead,
        alignment_lookaheads=schedule,
        trace=trace,
    )
    result = {"target_level": target_level, "lookahead": lookahead,
              "max_align": max_align, "low_macro": low_macro,
              "low_value": low_value, "actions": counted.moves,
              "high_macros": high_macros,
              "level": counted.levels_completed, "trace": trace}
    if os.environ.get("PRINT_PATH") == "1":
        result["path"] = counted.path
    print("PARAM_RESULT", result)


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
