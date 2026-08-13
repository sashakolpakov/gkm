"""Find a shorter key-only path to one known controller milestone."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (_bridge_carrier_state,
                  solve_parallel_wrapped_bridge_carrier_peg_solitaire)
from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


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


def state(env):
    return int(env.levels_completed), _bridge_carrier_state(env.frame())


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
    desired = int(os.environ.get("OPT_LEVEL", "8"))
    wanted_stage = int(os.environ.get("OPT_STAGE", "11"))
    max_states = int(os.environ.get("OPT_STATES", "1000"))
    key_actions = ((1, 2, 3, 4, 7)
                   if os.environ.get("OPT_INCLUDE_RESET") == "1"
                   else (1, 2, 3, 4))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current

    node = entry
    if desired == 7:
        recorder = Recorder(entry.clone())
        solve_parallel_wrapped_bridge_carrier_peg_solitaire(recorder)
        groups = split(tuple(recorder.path))
    else:
        groups = split(campaign[start:end])
    for stage, (known_keys, clicks) in enumerate(groups):
        if stage == wanted_stage:
            root = node.clone()
            target_node = node.clone()
            for action in known_keys:
                safe_step(target_node, action)
            target = state(target_node)
            depth_bound = len(known_keys) - 1
            if os.environ.get("OPT_UNDO") == "1":
                live = root.clone()
                root_state = state(live)
                adjacency = {}
                best_depth = {root_state: 0}

                def visit(current_state, depth):
                    adjacency.setdefault(current_state, {})
                    if depth >= depth_bound or len(best_depth) >= max_states:
                        return
                    for action in key_actions:
                        before_frame = arr(live.frame())[1:, :].tobytes()
                        safe_step(live, action)
                        child_state = state(live)
                        adjacency[current_state][action] = child_state
                        if arr(live.frame())[1:, :].tobytes() == before_frame:
                            continue
                        child_depth = depth + 1
                        if child_depth < best_depth.get(child_state, 10 ** 9):
                            best_depth[child_state] = child_depth
                            visit(child_state, child_depth)
                        safe_step(live, 7)
                        if state(live) != current_state:
                            raise AssertionError(("undo mismatch", action, depth))

                try:
                    visit(root_state, 0)
                except Exception as exc:
                    print("undo_error", desired, stage, repr(exc),
                          len(best_depth), flush=True)
                    return
                graph_queue = deque([(root_state, ())])
                graph_seen = {root_state}
                while graph_queue:
                    current_state, path = graph_queue.popleft()
                    for action, child_state in adjacency.get(
                            current_state, {}).items():
                        child_path = path + (action,)
                        if child_state == target:
                            print("shortcut", desired, stage, known_keys,
                                  child_path, len(best_depth),
                                  len(graph_seen), flush=True)
                            return
                        if child_state not in graph_seen:
                            graph_seen.add(child_state)
                            graph_queue.append((child_state, child_path))
                print("none", desired, stage, len(known_keys),
                      len(best_depth), len(graph_seen), flush=True)
                return
            queue = deque([(root, ())])
            seen = {state(root)}
            expanded = 0
            while queue and len(seen) <= max_states:
                current, path = queue.popleft()
                expanded += 1
                if len(path) >= depth_bound:
                    continue
                for action in key_actions:
                    child = current.clone()
                    safe_step(child, action)
                    child_path = path + (action,)
                    child_state = state(child)
                    if child_state == target:
                        print("shortcut", desired, stage, known_keys,
                              child_path, len(seen), expanded, flush=True)
                        return
                    if child_state in seen:
                        continue
                    seen.add(child_state)
                    queue.append((child, child_path))
            print("none", desired, stage, len(known_keys), len(seen),
                  expanded, flush=True)
            return
        for action in known_keys:
            safe_step(node, action)
        for action in clicks:
            safe_step(node, action)


arena.run_program("lf52", probe)
