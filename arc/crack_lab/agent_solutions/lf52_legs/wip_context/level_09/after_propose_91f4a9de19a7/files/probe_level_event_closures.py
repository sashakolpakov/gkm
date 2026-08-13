"""Enumerate key-reachable peg events at each milestone of a known level path."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import connected_components, safe_step


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


def generic_moves(frame):
    """Visible peg/bridge jumps, including movable color-8 bridges."""
    blobs = connected_components(frame, colors=(1, 8, 12, 14, 15))
    slots = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    slots |= {
        (blob.bbox[0] - 1, blob.bbox[1] - 1) for blob in blobs
        if blob.color == 1 and blob.size == (2, 2) and blob.area == 4
    }
    carriers = {
        ((blob.bbox[0] - 1, blob.bbox[1] - 1)
         if blob.size == (2, 2) else blob.top_left)
        for blob in blobs
        if blob.color == 12 and blob.size in ((2, 2), (4, 4))
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    movable = {
        blob.top_left for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    destinations = slots | carriers
    occupied = pegs | movable
    moves = []
    for kind, sources in (("peg", pegs), ("movable", movable)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if destination not in destinations or destination in occupied:
                    continue
                if kind == "movable":
                    legal = midpoint in pegs
                else:
                    legal = midpoint in pegs | movable | fixed
                if legal:
                    moves.append((kind, source, destination))
    return tuple(moves)


def closure(root, max_states, max_depth):
    queue = deque([(root.clone(), ())])
    seen = {_bridge_carrier_state(root.frame())}
    events = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        for move in set(_bridge_carrier_moves(node.frame())) | set(
                generic_moves(node.frame())):
            events.setdefault(move, path)
        if len(path) >= max_depth:
            continue
        actions = (1, 2, 3, 4, 7) if os.environ.get("OPT_RESET") == "1" else (1, 2, 3, 4)
        for action in actions:
            child = node.clone()
            safe_step(child, action)
            state = _bridge_carrier_state(child.frame())
            if state in seen:
                continue
            seen.add(state)
            queue.append((child, path + (action,)))
    return len(seen), events


def undo_closure(root, max_states, max_depth):
    """Traverse carrier controls in-place; action 7 restores each key step."""
    node = root.clone()
    root_state = _bridge_carrier_state(node.frame())
    states = {root_state}
    adjacency = {}
    state_events = {}

    def visit(state, depth):
        state_events[state] = tuple(
            set(_bridge_carrier_moves(node.frame()))
            | set(generic_moves(node.frame()))
        )
        adjacency.setdefault(state, {})
        if depth >= max_depth:
            return
        for action in (1, 2, 3, 4):
            safe_step(node, action)
            child = _bridge_carrier_state(node.frame())
            adjacency[state][action] = child
            is_new = child not in states and len(states) < max_states
            if is_new:
                states.add(child)
                visit(child, depth + 1)
            safe_step(node, 7)
            restored = _bridge_carrier_state(node.frame())
            if restored != state:
                raise AssertionError(("key undo mismatch", action, depth))

    visit(root_state, 0)
    queue = deque([(root_state, ())])
    reached = {root_state}
    events = {}
    while queue:
        state, path = queue.popleft()
        for move in state_events.get(state, ()):
            events.setdefault(move, path)
        for action, child in adjacency.get(state, {}).items():
            if child not in reached:
                reached.add(child)
                queue.append((child, path + (action,)))
    return len(states), events


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    max_states = int(os.environ.get("OPT_STATES", "350"))
    max_depth = int(os.environ.get("OPT_DEPTH", "16"))
    first_stage = int(os.environ.get("OPT_FIRST_STAGE", "0"))
    last_stage = int(os.environ.get("OPT_LAST_STAGE", "1000"))
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
    for stage, (known_keys, clicks) in enumerate(split(campaign[start:end])):
        if first_stage <= stage <= last_stage:
            closure_fn = (undo_closure if os.environ.get("OPT_UNDO") == "1"
                          else closure)
            effective_depth = max_depth
            if os.environ.get("OPT_ROUTE_DEPTH") == "1":
                effective_depth = min(
                    max_depth,
                    len(known_keys)
                    + int(os.environ.get("OPT_LOOKAHEAD", "0")),
                )
            states, events = closure_fn(node, max_states, effective_depth)
            ordered = tuple(sorted(
                (len(path), move, path) for move, path in events.items()
            ))
            print("stage", stage, "known", known_keys, "states", states,
                  "events", ordered, flush=True)
        for action in known_keys:
            safe_step(node, action)
        for action in clicks:
            safe_step(node, action)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
