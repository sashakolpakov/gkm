"""Find shorter key-only alignments for the reproduced level-7 path."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def frame_key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def split_path(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        action = normalize(path[index])
        if isinstance(action, int):
            keys.append(action)
            index += 1
            continue
        if index + 1 >= len(path):
            raise ValueError("unpaired coordinate action")
        second = normalize(path[index + 1])
        groups.append((tuple(keys), (action, second)))
        keys = []
        index += 2
    if keys:
        groups.append((tuple(keys), ()))
    return tuple(groups)


def shortest_keys(root, target, max_depth, max_states=3000):
    if frame_key(root) == target:
        return (), 1
    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_path = path + (action,)
            child_key = frame_key(child)
            if child_key == target:
                return child_path, len(seen)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    return None, len(seen)


def shortest_keys_bidirectional(root, target_env, max_depth,
                                max_states=50000):
    """Meet in the middle, using the observed directional inverse actions."""
    target = frame_key(target_env)
    if frame_key(root) == target:
        return (), 1
    inverse = {1: 2, 2: 1, 3: 4, 4: 3}
    forward_depth = max_depth // 2
    backward_depth = max_depth - forward_depth
    forward = {frame_key(root): ()}
    frontier = (root.clone(),)
    for _ in range(forward_depth):
        children = []
        for node in frontier:
            prefix = forward[frame_key(node)]
            for action in (1, 2, 3, 4):
                child = node.clone()
                safe_step(child, action)
                child_key = frame_key(child)
                if child_key in forward:
                    continue
                forward[child_key] = prefix + (action,)
                children.append(child)
                if len(forward) >= max_states:
                    return None, len(forward)
        frontier = tuple(children)

    backward = {target: ()}
    frontier = (target_env.clone(),)
    for depth in range(backward_depth + 1):
        children = []
        for node in frontier:
            node_key = frame_key(node)
            reverse_path = backward[node_key]
            if node_key in forward:
                candidate = forward[node_key] + tuple(
                    inverse[action] for action in reversed(reverse_path)
                )
                check = root.clone()
                for action in candidate:
                    safe_step(check, action)
                if frame_key(check) == target:
                    return candidate, len(forward) + len(backward)
            if depth >= backward_depth:
                continue
            for action in (1, 2, 3, 4):
                child = node.clone()
                safe_step(child, action)
                child_key = frame_key(child)
                if child_key in backward:
                    continue
                backward[child_key] = reverse_path + (action,)
                children.append(child)
                if len(forward) + len(backward) >= max_states:
                    return None, len(forward) + len(backward)
        frontier = tuple(children)
    return None, len(forward) + len(backward)


def probe(env):
    desired_level = int(os.environ.get("OPT_LEVEL", "7"))
    max_states = int(os.environ.get("OPT_STATES", "3000"))
    start_group = int(os.environ.get("OPT_START_GROUP", "0"))
    end_group = int(os.environ.get("OPT_END_GROUP", "1000000"))
    bidirectional = os.environ.get("OPT_BIDIR") == "1"
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    path = tuple(normalize(action) for action in checkpoint["final_path"])

    entry = None
    start_index = None
    end_index = None
    prior_level = int(env.levels_completed)
    for index, action in enumerate(path):
        safe_step(env, action)
        level = int(env.levels_completed)
        if prior_level < desired_level - 1 <= level:
            entry = env.clone()
            start_index = index + 1
        if prior_level < desired_level <= level:
            end_index = index + 1
            break
        prior_level = level
    level_path = path[start_index:end_index]
    groups = split_path(level_path)
    print("boundary", start_index, end_index, len(level_path),
          tuple(len(keys) for keys, _ in groups), flush=True)

    reference = entry.clone()
    optimized = entry.clone()
    chosen_groups = []
    for group_index, (original_keys, clicks) in enumerate(groups):
        for action in original_keys:
            safe_step(reference, action)
        target = frame_key(reference)
        if start_group <= group_index <= end_group:
            if bidirectional:
                replacement, states = shortest_keys_bidirectional(
                    optimized, reference,
                    max(0, len(original_keys) - 1), max_states
                )
            else:
                replacement, states = shortest_keys(
                    optimized, target,
                    max(0, len(original_keys) - 1), max_states
                )
        else:
            replacement, states = None, 0
        if replacement is None:
            replacement = original_keys
        for action in replacement:
            safe_step(optimized, action)
        same_before = frame_key(optimized) == target
        if not same_before:
            raise AssertionError((group_index, "pre-click mismatch"))
        for action in clicks:
            safe_step(reference, action)
            safe_step(optimized, action)
        same_after = frame_key(optimized) == frame_key(reference)
        if not same_after:
            raise AssertionError((group_index, "post-click mismatch"))
        chosen_groups.append((replacement, clicks))
        print("group", group_index, len(original_keys), original_keys,
              "=>", len(replacement), replacement, "states", states,
              "level", int(optimized.levels_completed), flush=True)

    optimized_path = tuple(
        action for keys, clicks in chosen_groups for action in keys + clicks
    )
    print("result", len(level_path), len(optimized_path),
          int(optimized.levels_completed), flush=True)
    print("groups", tuple(chosen_groups), flush=True)


arena.run_program("lf52", probe)
