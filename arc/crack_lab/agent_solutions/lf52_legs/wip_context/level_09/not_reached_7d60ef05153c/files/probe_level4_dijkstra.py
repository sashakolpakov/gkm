"""Bounded action-cost search for a shorter verified level-4 route."""

from heapq import heappop, heappush
import json
import os

import gkm_try

from legs import (
    _bridge_carrier_moves,
    _bridge_carrier_state,
    solve_bridge_carrier_peg_solitaire,
)
from perception import arr, safe_step


LEVEL_START = 87
MAX_STATES = int(os.environ.get("MAX_STATES", "12000"))


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


def key(node):
    return arr(node.frame())[1:, :].tobytes()


def apply(node, action):
    safe_step(node, action)


def reconstruct(parent, cursor):
    actions = []
    while parent[cursor] is not None:
        cursor, macro = parent[cursor]
        actions.extend(reversed(macro))
    actions.reverse()
    return actions


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    base_level = int(env.levels_completed)

    recorder = Recorder(env.clone())
    solve_bridge_carrier_peg_solitaire(recorder, reverse_choices=True)
    upper_path = list(recorder.actions)
    upper = len(upper_path)

    known = {}
    trace = env.clone()
    known[key(trace)] = tuple(upper_path)
    for index, action in enumerate(upper_path):
        apply(trace, action)
        known.setdefault(key(trace), tuple(upper_path[index + 1:]))

    root = env.clone(); root_key = key(root)
    distance = {root_key: 0}; parent = {root_key: None}
    nodes = {root_key: root}
    queue = [(0, 0, root_key)]
    serial = 0; popped = 0; best_path = upper_path

    while queue and popped < MAX_STATES:
        cost, _, state_key = heappop(queue)
        if cost != distance.get(state_key):
            continue
        node = nodes.pop(state_key)
        popped += 1
        suffix = known.get(state_key)
        if suffix is not None and cost + len(suffix) < upper:
            upper = cost + len(suffix)
            best_path = reconstruct(parent, state_key) + list(suffix)
            print("L4_DIJKSTRA_BOUND", popped, len(distance), upper, flush=True)
        if cost >= upper - 1:
            continue

        macros = [((action,), 1) for action in (1, 2, 3, 4)]
        macros += [(
            (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            ),
            2,
        ) for _, source, destination in _bridge_carrier_moves(node.frame())]
        for macro, weight in macros:
            child = node.clone()
            for action in macro:
                apply(child, action)
            child_cost = cost + weight
            child_key = key(child)
            if child_key == state_key or child_cost >= distance.get(child_key, upper):
                continue
            distance[child_key] = child_cost
            parent[child_key] = state_key, macro
            if int(child.levels_completed) > base_level:
                if child_cost < upper:
                    upper = child_cost
                    best_path = reconstruct(parent, child_key)
                    print("L4_DIJKSTRA_GOAL", popped, len(distance), upper, flush=True)
                continue
            nodes[child_key] = child
            serial += 1; heappush(queue, (child_cost, serial, child_key))

    replay = env.clone()
    for action in best_path:
        apply(replay, action)
    print(
        "L4_DIJKSTRA", popped, len(distance), len(best_path),
        int(replay.levels_completed), bool(queue), best_path,
    )


gkm_try.A.run_program("lf52", probe)
