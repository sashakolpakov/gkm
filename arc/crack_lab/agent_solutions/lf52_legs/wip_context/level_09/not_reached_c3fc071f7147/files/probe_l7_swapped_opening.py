"""Test the alternate first level-7 carrier exit and locate its handoffs."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import safe_step
from probe_level_event_closures import generic_moves


LOAD_PEG = (3, 3, 1, 1, 3, 3, 3)
ALT_UNLOAD_PEG = (4, 4, 4, 2, 2, 4, 4, 4, 2)


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def closure(root, target=None, max_states=80, max_depth=18):
    queue = deque([(root.clone(), ())])
    seen = {_bridge_carrier_state(root.frame())}
    hits = []
    events = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        moves = generic_moves(node.frame())
        for item in moves:
            events.setdefault(item, path)
        if target is not None and any(item[1:] == target for item in moves):
            hits.append((len(path), path))
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            state = _bridge_carrier_state(child.frame())
            if state in seen:
                continue
            seen.add(state)
            queue.append((child, path + (action,)))
    return tuple(sorted(hits)), len(seen), tuple(sorted(
        (len(path), move, path) for move, path in events.items()
    ))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 6 <= current:
            break
        prior = current

    node = env.clone()
    for action in LOAD_PEG:
        safe_step(node, action)
    move(node, (12, 6), (24, 6))
    for action in ALT_UNLOAD_PEG:
        safe_step(node, action)
    move(node, (42, 42), (54, 42))
    print("swapped_first", int(node.levels_completed), flush=True)

    hits, states, events = closure(node)
    print("swapped_closure", states, events, flush=True)


arena.run_program("lf52", probe)
