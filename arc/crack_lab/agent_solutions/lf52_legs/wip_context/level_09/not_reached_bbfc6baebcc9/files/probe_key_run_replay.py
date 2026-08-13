"""Path-only BFS for a shorter reproduced carrier-key milestone."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
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


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    wanted_stage = int(os.environ.get("OPT_STAGE", "3"))
    max_states = int(os.environ.get("OPT_STATES", "1000"))
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

    groups = split(campaign[start:end])
    root = entry
    for stage, (known_keys, clicks) in enumerate(groups):
        if stage == wanted_stage:
            target_node = root.clone()
            for action in known_keys:
                safe_step(target_node, action)
            target = _bridge_carrier_state(target_node.frame())
            depth_bound = len(known_keys) - 1
            break
        for action in known_keys + clicks:
            safe_step(root, action)

    root_state = _bridge_carrier_state(root.frame())
    queue = deque([((), root_state)])
    seen = {root_state}
    expanded = 0
    while queue and len(seen) <= max_states:
        path, expected = queue.popleft()
        if len(path) >= depth_bound:
            continue
        node = root.clone()
        for action in path:
            safe_step(node, action)
        if _bridge_carrier_state(node.frame()) != expected:
            raise AssertionError("reconstruction mismatch")
        before_key = key(node)
        expanded += 1
        if expanded % 100 == 0:
            print("replay_progress", expanded, len(seen), len(path),
                  flush=True)
        for action in (1, 2, 3, 4):
            safe_step(node, action)
            child_key = key(node)
            if child_key == before_key:
                continue
            child_state = _bridge_carrier_state(node.frame())
            child_path = path + (action,)
            if child_state == target:
                print("replay_shortcut", desired, wanted_stage,
                      known_keys, child_path, len(seen), expanded,
                      flush=True)
                return
            if child_state not in seen:
                seen.add(child_state)
                queue.append((child_path, child_state))
            safe_step(node, 7)
            if key(node) != before_key:
                raise AssertionError(("undo mismatch", path, action))
    print("replay_none", desired, wanted_stage, len(known_keys),
          len(seen), expanded, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    if error:
        print("replay_worker_error", repr(error), flush=True)
