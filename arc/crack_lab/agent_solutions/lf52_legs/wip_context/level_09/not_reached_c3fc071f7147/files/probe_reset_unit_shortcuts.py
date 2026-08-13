"""Replace contiguous route-unit blocks by one reset action and replay."""

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


def units(path):
    result = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            end = index + 1
            while end < len(path) and isinstance(path[end], int):
                end += 1
            result.append(tuple(path[index:end]))
            index = end
        else:
            result.append((path[index], path[index + 1]))
            index += 2
    return tuple(result)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = env.clone() if desired == 1 else None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current

    recorder = Recorder(entry.clone())
    SOLVERS[desired](recorder)
    path = tuple(recorder.path)
    route_units = units(path)
    if os.environ.get("OPT_STRUCTURE") == "1":
        print("route_structure", desired, len(path),
              tuple(len(unit) for unit in route_units), flush=True)
        return
    prefix_costs = [0]
    prefix_nodes = [entry.clone()]
    node = entry.clone()
    for unit in route_units:
        for action in unit:
            safe_step(node, action)
        prefix_costs.append(prefix_costs[-1] + len(unit))
        prefix_nodes.append(node.clone())

    if os.environ.get("OPT_MATCH_ONLY") == "1":
        prefix_keys = [
            (int(node.levels_completed), node.frame()[1:, :].tobytes())
            for node in prefix_nodes
        ]
        matches = []
        for left, node in enumerate(prefix_nodes[:-1]):
            child = node.clone()
            safe_step(child, 7)
            child_key = (
                int(child.levels_completed), child.frame()[1:, :].tobytes()
            )
            for right in range(left + 1, len(prefix_nodes)):
                if (
                    child_key == prefix_keys[right]
                    and prefix_costs[left] + 1 < prefix_costs[right]
                ):
                    matches.append((
                        prefix_costs[left] + 1
                        + len(path) - prefix_costs[right],
                        left,
                        right,
                    ))
        print("reset_matches", desired, len(path), len(route_units),
              tuple(sorted(matches)), flush=True)
        return

    wins = []
    left_start = int(os.environ.get("OPT_LEFT_START", "0"))
    left_end = min(len(route_units),
                   int(os.environ.get("OPT_LEFT_END", str(len(route_units)))))
    for left in range(left_start, left_end):
        for right in range(left + 1, len(route_units) + 1):
            candidate_bound = (prefix_costs[left] + 1
                               + len(path) - prefix_costs[right])
            if candidate_bound >= len(path):
                continue
            child = prefix_nodes[left].clone()
            safe_step(child, 7)
            used = prefix_costs[left] + 1
            won = int(child.levels_completed) > int(entry.levels_completed)
            for unit in route_units[right:]:
                if won:
                    break
                for action in unit:
                    safe_step(child, action)
                    used += 1
                    if int(child.levels_completed) > int(entry.levels_completed):
                        won = True
                        break
            if won and used < len(path):
                wins.append((used, left, right,
                             tuple(route_units[left:right])))
        print("progress", desired, left + 1, len(route_units), len(wins),
              flush=True)
    print("reset_shortcuts", desired, len(path), len(route_units),
          tuple(sorted(wins)[:30]), flush=True)


arena.run_program("lf52", probe)
