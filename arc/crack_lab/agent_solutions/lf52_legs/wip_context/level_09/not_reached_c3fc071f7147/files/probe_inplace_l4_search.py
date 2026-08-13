"""Exact in-place search for a shorter compact level-4 suffix."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state, solve_compact_bridge_carrier_peg_solitaire
from perception import arr, safe_step
from probe_key_neighborhood_events import generic_moves


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


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    return tuple(groups)


def frame_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 3 <= current:
            entry = env.clone()
            break
        prior = current

    recorder = Recorder(entry.clone())
    solve_compact_bridge_carrier_peg_solitaire(recorder)
    groups = split(tuple(recorder.path))
    start_stage = int(os.environ.get("OPT_START_STAGE", "10"))
    node = entry.clone()
    prefix = []
    for keys, clicks in groups[:start_stage]:
        for action in keys + clicks:
            safe_step(node, action)
            prefix.append(action)
    baseline = sum(len(keys) + 2 for keys, _ in groups[start_stage:])
    bound = int(os.environ.get("OPT_BOUND", str(baseline - 1)))
    max_expanded = int(os.environ.get("OPT_STATES", "5000"))
    max_steps = int(os.environ.get("OPT_STEPS", "50000"))
    max_key_run = int(os.environ.get("OPT_KEY_RUN", "12"))
    base_level = int(node.levels_completed)
    best = {frame_key(node): 0}
    path = []
    solution = None
    expanded = 0
    steps = 0

    def step(action):
        nonlocal steps
        safe_step(node, action)
        steps += 1

    def restore(target, count):
        for _ in range(count):
            step(7)
        restored = frame_key(node)
        if restored != target:
            raise AssertionError(("restore", count, restored[0], target[0]))

    def visit(cost, key_run):
        nonlocal expanded, solution
        if (
            solution is not None
            or expanded >= max_expanded
            or steps >= max_steps
            or cost >= bound
        ):
            return
        expanded += 1
        if expanded % 250 == 0:
            print("inplace_progress", expanded, steps, cost, len(best),
                  flush=True)
        before = frame_key(node)
        moves = list(generic_moves(node.frame()))
        moves.sort(key=lambda move: (0 if move[0] == "peg" else 1, move))
        for _, source, destination in moves:
            if cost + 2 > bound:
                continue
            source_action = (6, source[1] + 1, source[0] + 1)
            destination_action = (
                6, destination[1] + 1, destination[0] + 1,
            )
            step(source_action)
            selected = frame_key(node)
            if selected == before:
                continue
            step(destination_action)
            after = frame_key(node)
            selection = _bridge_carrier_state(node.frame())[5]
            valid = after != selected and selection is None
            if valid:
                path.extend((source_action, destination_action))
                if int(node.levels_completed) > base_level:
                    solution = tuple(path)
                    return
                child_cost = cost + 2
                if child_cost < best.get(after, 10 ** 9):
                    best[after] = child_cost
                    visit(child_cost, 0)
                del path[-2:]
                if solution is not None:
                    return
                restore(before, 2)
            else:
                restore(before, 1 if after == selected else 2)

        if key_run >= max_key_run:
            return
        for action in (1, 2, 3, 4):
            if cost + 1 > bound:
                continue
            step(action)
            after = frame_key(node)
            if after == before:
                continue
            path.append(action)
            child_cost = cost + 1
            if child_cost < best.get(after, 10 ** 9):
                best[after] = child_cost
                visit(child_cost, key_run + 1)
            path.pop()
            if solution is not None:
                return
            restore(before, 1)

    try:
        visit(0, 0)
    except Exception as error:
        print("inplace_error", repr(error), expanded, steps, tuple(path),
              flush=True)
        raise
    print("inplace_result", start_stage, baseline, bound, expanded, steps,
          None if solution is None else len(solution), solution, flush=True)


arena.run_program("lf52", probe)
