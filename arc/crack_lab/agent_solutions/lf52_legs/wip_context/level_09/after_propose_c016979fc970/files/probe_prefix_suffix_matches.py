"""Match partial fixed-leg prefixes to cheaper native suffix entry states."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_state,
    _movable_bridge_board,
    solve_direct_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_multi_bridge_wrapped_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
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
    solve_wrapped_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_multi_bridge_wrapped_carrier_peg_solitaire,
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


def keys(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    full = frame.tobytes()
    bridge = _bridge_carrier_state(env.frame())
    movable = tuple(
        frozenset(part) for part in _movable_bridge_board(env.frame())
    )
    abstract = (bridge, movable)
    relaxed = (
        bridge[1], bridge[2], bridge[3], bridge[4],
        movable[1], movable[2], movable[3],
    )
    return full, abstract, relaxed


def trace(entry, units):
    node = entry.clone()
    states = [(0, 0, keys(node))]
    action_cost = 0
    for unit_index, unit in enumerate(units, 1):
        for action in unit:
            play(node, action)
            action_cost += 1
        states.append((unit_index, action_cost, keys(node)))
        if node.levels_completed > entry.levels_completed:
            break
    return states


def native_actions(level, campaign):
    if level == 5:
        return campaign[149:238]
    with open(NATIVE_FILES[level]) as native_file:
        return json.load(native_file)


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
    for level in range(5, 10):
        native = native_actions(level, campaign)
        native_units = macro_units(native)
        native_trace = trace(entries[level], native_units)
        for staging_solver in STAGING:
            recorder = RecordingEnv(entries[level].clone())
            staging_solver(recorder)
            staging_units = macro_units(recorder.actions_taken)
            staging_trace = trace(entries[level], staging_units)
            candidates = set()
            for stage_unit, stage_cost, stage_keys in staging_trace:
                for native_unit, native_cost, native_keys in native_trace:
                    if stage_cost >= native_cost:
                        continue
                    match_kind = next((
                        name for name, index in (
                            ("full", 0), ("abstract", 1), ("relaxed", 2)
                        )
                        if stage_keys[index] == native_keys[index]
                    ), None)
                    if match_kind is not None:
                        candidates.add((
                            stage_unit, native_unit, match_kind,
                            stage_cost, native_cost,
                        ))
            for (
                stage_unit, native_unit, match_kind,
                stage_cost, native_cost,
            ) in sorted(candidates, key=lambda item: (item[3], -item[4])):
                trial = entries[level].clone()
                prefix = flatten(staging_units[:stage_unit])
                suffix = flatten(native_units[native_unit:])
                for action in prefix:
                    play(trial, action)
                executed = []
                for action in suffix:
                    play(trial, action)
                    executed.append(action)
                    if trial.levels_completed > level - 1:
                        break
                if trial.levels_completed > level - 1:
                    candidate = prefix + executed
                    if len(candidate) >= len(native):
                        continue
                    winners.append((level, staging_solver.__name__, candidate))
                    print("WIN", {
                        "level": level,
                        "staging": staging_solver.__name__,
                        "match": match_kind,
                        "costs": (stage_cost, native_cost),
                        "native": len(native),
                        "candidate": len(candidate),
                    }, flush=True)
            print("MATCHES", {
                "level": level,
                "staging": staging_solver.__name__,
                "candidates": len(candidates),
            }, flush=True)
    for level, staging_name, candidate in winners:
        filename = f"level{level}_partial_splice_{len(candidate)}.json"
        with open(filename, "w") as candidate_file:
            json.dump(candidate, candidate_file, indent=2)
            candidate_file.write("\n")
        print("SAVED", {
            "level": level,
            "staging": staging_name,
            "file": filename,
        })
    print("RESULT", {"winners": len(winners)})


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
