"""Find shorter level-7 carrier-key paths while preserving each visible state."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


TARGET_LEVEL = int(os.environ.get("SHORTCUT_LEVEL", "7"))
ENTRY_INDEX = int(os.environ.get("SHORTCUT_ENTRY", "331"))
EXIT_INDEX = int(os.environ.get("SHORTCUT_EXIT", "476"))
KEY_MODE = os.environ.get("SHORTCUT_KEY", "frame")
KEY_ACTIONS = (1, 2, 3, 4)
MAX_STATES_PER_SEGMENT = 12000
OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def physical_key(env):
    if KEY_MODE.startswith("bridge"):
        state = _bridge_carrier_state(env.frame())
        return state[1:5] if KEY_MODE == "bridge_compact" else state
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def split_segments(actions):
    segments = []
    index = 0
    while index < len(actions):
        keys = []
        while index < len(actions) and isinstance(actions[index], int):
            keys.append(actions[index])
            index += 1
        clicks = []
        while index < len(actions) and not isinstance(actions[index], int):
            clicks.append(actions[index])
            index += 1
        segments.append((keys, clicks))
    return segments


def search_side(root, radius, backward=False):
    queue = deque([(root.clone(), ())])
    seen = {physical_key(root): (root.clone(), ())}
    while queue and len(seen) < MAX_STATES_PER_SEGMENT:
        state, path = queue.popleft()
        if len(path) >= radius:
            continue
        for action in KEY_ACTIONS:
            child = state.clone()
            child.step(action)
            key = physical_key(child)
            if key in seen:
                continue
            child_path = (
                (OPPOSITE[action],) + path
                if backward else path + (action,)
            )
            seen[key] = (child, child_path)
            queue.append((child, child_path))
    return seen


def shortest_key_path(entry, target, depth_limit):
    target_key = physical_key(target)
    if physical_key(entry) == target_key:
        return [], 2
    forward_radius = depth_limit // 2
    backward_radius = depth_limit - forward_radius
    forward = search_side(entry, forward_radius)
    backward = search_side(target, backward_radius, backward=True)
    candidates = []
    for key in forward.keys() & backward.keys():
        path = forward[key][1] + backward[key][1]
        if len(path) <= depth_limit:
            candidates.append(path)
    for path in sorted(candidates, key=len):
        node = entry.clone()
        play(node, path)
        if physical_key(node) == target_key:
            return list(path), len(forward) + len(backward)
    return None, len(forward) + len(backward)


def play(env, actions):
    for action in actions:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    play(env, campaign[:ENTRY_INDEX])
    entry_level = env.levels_completed
    original = campaign[ENTRY_INDEX:EXIT_INDEX]
    candidate = []
    reports = []
    for number, (keys, clicks) in enumerate(split_segments(original), 1):
        target = env.clone()
        play(target, keys)
        if keys:
            shortcut, states = shortest_key_path(env, target, len(keys) - 1)
        else:
            shortcut, states = [], 1
        chosen = keys if shortcut is None else shortcut
        play(env, chosen)
        matched = physical_key(env) == physical_key(target)
        play(env, clicks)
        candidate.extend(chosen)
        candidate.extend(clicks)
        reports.append({
            "segment": number,
            "keys": (len(keys), len(chosen)),
            "states": states,
            "matched": matched,
        })
        print("SEGMENT", reports[-1], flush=True)
    filename = f"level{TARGET_LEVEL}_key_shortcuts_candidate.json"
    with open(filename, "w") as candidate_file:
        json.dump(candidate, candidate_file, indent=2)
        candidate_file.write("\n")
    print("RESULT", {
        "entry_level": entry_level,
        "target_level": TARGET_LEVEL,
        "levels": env.levels_completed,
        "original": len(original),
        "candidate": len(candidate),
        "saved": len(original) - len(candidate),
        "segments": reports,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
