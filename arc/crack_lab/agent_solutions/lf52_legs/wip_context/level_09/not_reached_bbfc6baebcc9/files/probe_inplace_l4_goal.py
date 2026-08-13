"""Branch-and-bound level-4 search using only verified public undo."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, solve_compact_bridge_carrier_peg_solitaire
from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


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
            result.append((path[index],))
            index += 1
        else:
            result.append((path[index], path[index + 1]))
            index += 2
    return tuple(result)


def macros(frame):
    moves = tuple(_bridge_carrier_moves(frame))
    result = []
    for kind, source, destination in moves:
        result.append((
            0 if kind == "capture" else 1,
            ((6, source[1] + 1, source[0] + 1),
             (6, destination[1] + 1, destination[0] + 1)),
        ))
    result.extend((2, (action,)) for action in (4, 3, 2, 1))
    return tuple(macro for _, macro in sorted(result, key=lambda item: item[0]))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < 3 <= current:
            entry = env.clone()
            break
        prior = current

    recorder = Recorder(entry.clone())
    solve_compact_bridge_carrier_peg_solitaire(recorder)
    route = units(tuple(recorder.path))
    reference = entry.clone()
    preferred = {}
    for macro in route:
        preferred[state_key(reference)] = macro
        for action in macro:
            safe_step(reference, action)

    node = entry.clone()
    for action in route[0]:
        safe_step(node, action)
    base_level = int(node.levels_completed)
    prefix = tuple(route[0])
    bound = int(os.environ.get("OPT_COST", "49")) - len(prefix)
    max_expanded = int(os.environ.get("OPT_STATES", "2000"))
    best = {state_key(node): 0}
    path = []
    expanded = 0
    solution = None
    failed = None

    def visit(cost):
        nonlocal expanded, solution, failed
        if solution is not None or failed is not None or expanded >= max_expanded:
            return
        expanded += 1
        if expanded % 250 == 0:
            print("inplace_progress", expanded, len(best), cost, len(path),
                  flush=True)
        before = state_key(node)
        options = list(macros(node.frame()))
        wanted = preferred.get(before)
        if wanted in options:
            options.remove(wanted)
            options.insert(0, wanted)
        for macro in options:
            child_cost = cost + len(macro)
            if child_cost > bound:
                continue
            executed_steps = 0
            valid = True
            for action in macro:
                step_before = state_key(node)
                safe_step(node, action)
                executed_steps += 1
                if state_key(node) != step_before:
                    pass
                else:
                    valid = False
                    break
            after = state_key(node)
            if valid and int(node.levels_completed) > base_level:
                solution = prefix + tuple(path) + macro
                return
            if valid and after != before and child_cost < best.get(
                    after, 10 ** 9):
                best[after] = child_cost
                path.extend(macro)
                visit(child_cost)
                del path[-len(macro):]
            for _ in range(executed_steps):
                safe_step(node, 7)
            restored = state_key(node)
            if restored != before:
                failed = (cost, macro, executed_steps)
                return
            if solution is not None or expanded >= max_expanded:
                return

    visit(0)
    print("inplace_result", len(recorder.path), bound + len(prefix),
          expanded, len(best), failed,
          None if solution is None else len(solution), solution, flush=True)


arena.run_program("lf52", probe)
