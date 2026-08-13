"""Test crossing the initial level-7 peg/bridge carrier destinations."""

from collections import deque
import json

import gkm_try

from legs import _movable_bridge_board
from perception import arr, connected_components, safe_step


LEVEL_START = 331


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return tuple(sorted(carriers)), tuple(sorted(bridges)), tuple(sorted(pegs))


def moved(node, source, destination):
    trial = node.clone(); before = frame_key(trial)
    safe_step(trial, (6, source[1] + 1, source[0] + 1))
    safe_step(trial, (6, destination[1] + 1, destination[0] + 1))
    selected = any(
        blob.color == 3 and blob.area >= 4
        for blob in connected_components(trial.frame(), colors=(3,))
    )
    return not selected and frame_key(trial) != before


def plausible(node, source, destination):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    midpoint = (
        (source[0] + destination[0]) // 2,
        (source[1] + destination[1]) // 2,
    )
    return (
        source in bridges | pegs
        and destination in slots | carriers
        and destination not in bridges | pegs
        and midpoint in bridges | pegs | fixed
    )


def align(root, desired, max_states=500, max_depth=20):
    queue = deque([(root.clone(), ())]); seen = {frame_key(root)}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if plausible(node, *desired):
            return path, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (action,)))
    return None, len(seen)


def apply_move(node, desired, actions):
    source, destination = desired
    for action in (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    ):
        safe_step(node, action); actions.append(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    actions = []

    stages = (
        ((12, 6), (24, 6)),
        ((42, 42), (54, 42)),
        ((12, 42), (24, 42)),
        ((42, 12), (54, 12)),
    )
    trace = []
    for desired in stages:
        keys, searched = align(env, desired)
        trace.append((desired, None if keys is None else len(keys), searched, compact(env)))
        print("L7_CROSS_STAGE", trace[-1], flush=True)
        if keys is None:
            print("L7_CROSS", trace); return
        for action in keys:
            safe_step(env, action); actions.append(action)
        apply_move(env, desired, actions)

    crossed = (
        ((54, 12), (54, 24)),
        ((54, 24), (54, 36)),
        ((54, 36), (54, 48)),
        ((54, 42), (54, 54)),
        ((54, 4), (54, 16)),
    )
    for desired in crossed:
        if not moved(env, *desired):
            trace.append(("blocked", desired, compact(env)))
            break
        apply_move(env, desired, actions)
    else:
        desired = ((54, 10), (54, 22))
        keys, searched = align(env, desired)
        trace.append(("cross_load_peg", None if keys is None else len(keys), searched, compact(env)))
        if keys is not None:
            for action in keys:
                safe_step(env, action); actions.append(action)
            apply_move(env, desired, actions)
            bridge_advance = ((54, 16), (54, 28))
            if moved(env, *bridge_advance):
                apply_move(env, bridge_advance, actions)
                trace.append(("bridge_advance", compact(env)))
                relay = (
                    ((24, 34), (24, 46)),
                    ((54, 28), (42, 28)),
                    ((18, 46), (30, 46)),
                )
                for relay_move in relay:
                    relay_keys, relay_searched = align(env, relay_move)
                    trace.append((
                        "cross_relay", relay_move,
                        None if relay_keys is None else len(relay_keys),
                        relay_searched, compact(env),
                    ))
                    print("L7_CROSS_STAGE", trace[-1], flush=True)
                    if relay_keys is None:
                        break
                    for action in relay_keys:
                        safe_step(env, action); actions.append(action)
                    apply_move(env, relay_move, actions)
                else:
                    finish = (
                        ((24, 46), (36, 46)),
                        ((30, 46), (42, 46)),
                        ((36, 46), (36, 58)),
                    )
                    for finish_move in finish:
                        if not moved(env, *finish_move):
                            trace.append(("cross_finish_blocked", finish_move, compact(env)))
                            break
                        apply_move(env, finish_move, actions)
                    else:
                        trace.append(("cross_finish", compact(env)))
    print("L7_CROSS", len(actions), trace, compact(env), int(env.levels_completed))
    print("L7_CROSS_ACTIONS", actions)


gkm_try.A.run_program("lf52", probe)
