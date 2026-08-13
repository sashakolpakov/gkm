"""Reward-test shorter fixed-leg prefixes against every native macro suffix."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_direct_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
)


ENTRIES = {5: 149, 6: 238, 7: 331, 8: 476, 9: 544}
NATIVE_FILES = {
    6: "level6_greedy_macro_candidate.json",
    7: "level7_greedy_macro_candidate.json",
    8: "level8_greedy_macro_candidate.json",
    9: "level9_candidate_102.json",
}
STAGING = (
    solve_direct_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
)
TARGET_LEVELS = tuple(
    int(level)
    for level in os.environ.get("SPLICE_LEVELS", "5,6,7,8,9").split(",")
)


class RecordingEnv:
    def __init__(self, base):
        self.base = base
        self.actions_taken = []

    def step(self, *action):
        self.actions_taken.append(
            list(action) if len(action) > 1 else action[0]
        )
        return self.base.step(*action)

    def clone(self):
        return RecordingEnv(self.base.clone())

    def __getattr__(self, name):
        return getattr(self.base, name)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def macro_units(actions):
    units = []
    index = 0
    while index < len(actions):
        if isinstance(actions[index], list):
            units.append((actions[index], actions[index + 1]))
            index += 2
        else:
            units.append((actions[index],))
            index += 1
    return units


def flatten(units):
    return [action for unit in units for action in unit]


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    entries = {}
    node = env.clone()
    for index, action in enumerate(campaign):
        if index in ENTRIES.values():
            level = next(
                level for level, point in ENTRIES.items() if point == index
            )
            entries[level] = node.clone()
        play(node, action)
    entries[9] = node.clone()
    winners = []
    for level in TARGET_LEVELS:
        if level == 5:
            native = campaign[149:238]
        else:
            with open(NATIVE_FILES[level]) as native_file:
                native = json.load(native_file)
        native_units = macro_units(native)
        for staging in STAGING:
            staged = RecordingEnv(entries[level].clone())
            staging(staged)
            staging_actions = staged.actions_taken
            if staged.levels_completed > level - 1:
                continue
            for cut in range(len(native_units) + 1):
                suffix = flatten(native_units[cut:])
                trial = staged.base.clone()
                executed = []
                for action in suffix:
                    play(trial, action)
                    executed.append(action)
                    if trial.levels_completed > level - 1:
                        break
                if trial.levels_completed > level - 1:
                    candidate = staging_actions + executed
                    if len(candidate) >= len(native):
                        continue
                    winners.append((level, staging.__name__, cut, candidate))
                    print("WIN", {
                        "level": level,
                        "staging": staging.__name__,
                        "cut": cut,
                        "native": len(native),
                        "candidate": len(candidate),
                    }, flush=True)
    for level, staging_name, cut, candidate in winners:
        filename = f"level{level}_splice_{len(candidate)}.json"
        with open(filename, "w") as candidate_file:
            json.dump(candidate, candidate_file, indent=2)
            candidate_file.write("\n")
        print("SAVED", {
            "level": level,
            "staging": staging_name,
            "cut": cut,
            "file": filename,
        })
    print("RESULT", {
        "winners": len(winners),
        "best": {
            level: min(
                len(candidate)
                for winner_level, _, _, candidate in winners
                if winner_level == level
            )
            for level in sorted({winner[0] for winner in winners})
        },
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
