"""Bidirectionally shorten one selected level-7 carrier key run."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board


START = 331
END = 476
TARGET_RUN = int(os.environ.get("KEY_RUN", "8"))
INVERSE = {1: 2, 2: 1, 3: 4, 4: 3}
MAX_SIDE_STATES = 5500


def state_key(env):
    return tuple(frozenset(part) for part in _movable_bridge_board(env.frame()))


def groups(actions):
    out = []
    index = 0
    while index < len(actions):
        if isinstance(actions[index], int):
            start = index
            while index < len(actions) and isinstance(actions[index], int):
                index += 1
            out.append(actions[start:index])
        else:
            out.append(actions[index:index + 2])
            index += 2
    return out


def flatten(grouped):
    return [action for group in grouped for action in group]


def explore(root, max_depth):
    paths = {state_key(root): ()}
    queue = deque([(root.clone(), ())])
    while queue and len(paths) < MAX_SIDE_STATES:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = state_key(child)
            if child_key in paths:
                continue
            child_path = path + (action,)
            paths[child_key] = child_path
            queue.append((child, child_path))
    return paths


def replay(entry, actions):
    clone = entry.clone()
    for action in actions:
        clone.step(action)
    return clone


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:START]:
        env.step(action)
    entry = env.clone()
    grouped = groups(prefix[START:END])
    key_indices = [
        index for index, group in enumerate(grouped)
        if isinstance(group[0], int)
    ]
    group_index = key_indices[TARGET_RUN - 1]
    original = grouped[group_index]
    states = [entry.clone()]
    current = entry.clone()
    for group in grouped:
        for action in group:
            current.step(action)
        states.append(current.clone())
    start_state = states[group_index]
    target_state = states[group_index + 1]
    max_total = len(original) - 1
    forward_depth = max_total // 2
    backward_depth = max_total - forward_depth
    forward = explore(start_state, forward_depth)
    backward = explore(target_state, backward_depth)
    candidates = []
    for key in set(forward) & set(backward):
        prefix_path = forward[key]
        backward_path = backward[key]
        suffix_path = tuple(INVERSE[action] for action in reversed(backward_path))
        candidate = prefix_path + suffix_path
        if len(candidate) < len(original):
            candidates.append(candidate)
    accepted = None
    for candidate in sorted(set(candidates), key=lambda path: (len(path), path)):
        child = replay(start_state, candidate)
        if state_key(child) != state_key(target_state):
            continue
        candidate_groups = grouped[:group_index] + [list(candidate)] + grouped[group_index + 1:]
        candidate_actions = flatten(candidate_groups)
        final = replay(entry, candidate_actions)
        if final.levels_completed > entry.levels_completed:
            accepted = candidate
            filename = f"level7_keyrun{TARGET_RUN}_{len(candidate_actions)}.json"
            with open(filename, "w") as output_file:
                json.dump(candidate_actions, output_file, indent=2)
                output_file.write("\n")
            break
    print("RESULT", {
        "run": TARGET_RUN,
        "original": original,
        "forward_states": len(forward),
        "backward_states": len(backward),
        "intersections": len(candidates),
        "accepted": accepted,
        "level_actions": None if accepted is None else len(flatten(
            grouped[:group_index] + [list(accepted)] + grouped[group_index + 1:]
        )),
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
