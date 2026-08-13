"""Verify an alternate post-capture route with the established bridge leg."""

import json
from collections import deque

import gkm_try
from legs import _bridge_carrier_moves, solve_bridge_carrier_peg_solitaire
from perception import arr


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


def groups(segment):
    result = []; index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        result.append((tuple(keys), tuple(pair)))
    return result


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:149]: env.step(action)
    entry = env.clone(); level_groups = groups(path[149:238]); prefix = []
    for keys, pair in level_groups[:8]:
        for action in keys: env.step(action); prefix.append(action)
        for action in pair: env.step(*action); prefix.append(action)
    alternate = (1, (6, 46, 19), (6, 34, 19))
    for action in alternate:
        if isinstance(action, tuple): env.step(*action)
        else: env.step(action)
    frontier_root = env.clone(); queue = deque([(frontier_root.clone(), ())])
    seen = {arr(frontier_root.frame())[1:, :].tobytes()}; frontier = {}
    while queue and len(seen) <= 200:
        node, key_path = queue.popleft()
        for move in _bridge_carrier_moves(node.frame()): frontier.setdefault(move, key_path)
        if len(key_path) >= 16: continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action); child_key = arr(child.frame())[1:, :].tobytes()
            if child_key in seen: continue
            seen.add(child_key); queue.append((child, key_path + (action,)))
    print("ALTERNATIVE_FRONTIER", len(seen), tuple(sorted(frontier.items())))
    recorder = Recorder(env)
    solve_bridge_carrier_peg_solitaire(recorder, max_align_states=650)
    actions = prefix + list(alternate) + recorder.actions
    validation = entry.clone()
    for action in actions:
        if isinstance(action, tuple): validation.step(*action)
        else: validation.step(action)
        if validation.levels_completed >= 5: break
    print(
        "ALTERNATIVE_RESULT", len(prefix), len(recorder.actions), len(actions),
        recorder.levels_completed, validation.levels_completed >= 5, actions,
    )


gkm_try.A.run_program("lf52", probe)
