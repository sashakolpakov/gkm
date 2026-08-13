"""Delete whole action macros from a reproduced level path and reverify reward."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    solve_compact_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_repeated_frontier_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)
from perception import safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def units(path):
    out = []
    index = 0
    while index < len(path):
        action = path[index]
        if isinstance(action, int):
            out.append((action,))
            index += 1
        else:
            out.append((action, path[index + 1]))
            index += 2
    return out


def flatten(path_units):
    return tuple(action for unit in path_units for action in unit)


SOLVERS = {
    4: solve_compact_bridge_carrier_peg_solitaire,
    6: solve_wrapped_bridge_carrier_peg_solitaire,
    7: solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    8: solve_grid_wrapped_bridge_carrier_peg_solitaire,
    9: solve_repeated_frontier_bridge_carrier_peg_solitaire,
}


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.inner.step(action, *coordinates)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    max_tests = int(os.environ.get("OPT_TESTS", "80"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    start = None
    end = None
    entry = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            start = index + 1
            entry = env.clone()
            if os.environ.get("OPT_CURRENT") == "1":
                break
        if prior < desired <= current:
            end = index + 1
            break
        prior = current
    if os.environ.get("OPT_CURRENT") == "1":
        recorder = Recorder(entry.clone())
        SOLVERS[desired](recorder)
        current_units = units(tuple(recorder.path))
    else:
        level_path = campaign[start:end]
        if os.environ.get("OPT_ALT_STAGE3") == "1":
            if desired != 5:
                raise ValueError("stage-3 alternate belongs to level 5")
            groups = []
            keys = []
            index = 0
            while index < len(level_path):
                if isinstance(level_path[index], int):
                    keys.append(level_path[index])
                    index += 1
                else:
                    groups.append((tuple(keys),
                                   (level_path[index], level_path[index + 1])))
                    keys = []
                    index += 2
            groups[3] = (
                (2, 3, 2, 2, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 2),
                groups[3][1],
            )
            level_path = tuple(
                action for group_keys, clicks in groups
                for action in group_keys + clicks
            )
        if os.environ.get("OPT_L8_REALIGN") == "1":
            if desired != 8:
                raise ValueError("realigned route belongs to level 8")
            groups = []
            keys = []
            index = 0
            while index < len(level_path):
                if isinstance(level_path[index], int):
                    keys.append(level_path[index])
                    index += 1
                else:
                    groups.append((tuple(keys),
                                   (level_path[index], level_path[index + 1])))
                    keys = []
                    index += 2
            groups[0] = ((3, 3, 1, 1, 1), groups[0][1])
            groups[10] = ((1, 4), groups[10][1])
            level_path = tuple(
                action for group_keys, clicks in groups
                for action in group_keys + clicks
            )
        current_units = units(level_path)
    tests = 0

    def wins(candidate):
        node = entry.clone()
        for action in flatten(candidate):
            safe_step(node, action)
            if int(node.levels_completed) >= desired:
                return True
        return False

    print("start", desired, len(current_units), len(flatten(current_units)),
          flush=True)
    chunks = tuple(int(value) for value in
                   os.environ.get("OPT_CHUNKS", "64,32,16,8,4,2,1").split(","))
    for chunk in chunks:
        index = int(os.environ.get("OPT_START_INDEX", "0"))
        stop_limit = min(
            len(current_units),
            int(os.environ.get("OPT_END_INDEX", str(len(current_units)))),
        )
        while index < stop_limit and tests < max_tests:
            stop = min(len(current_units), index + chunk)
            candidate = current_units[:index] + current_units[stop:]
            tests += 1
            if wins(candidate):
                print("delete", chunk, index, stop,
                      len(flatten(current_units)), "=>", len(flatten(candidate)),
                      flush=True)
                current_units = candidate
            else:
                index += chunk
        print("pass", chunk, tests, len(flatten(current_units)), flush=True)
        if tests >= max_tests:
            break
    print("result", tests, len(flatten(current_units)),
          wins(current_units), tuple(current_units), flush=True)


arena.run_program("lf52", probe)
