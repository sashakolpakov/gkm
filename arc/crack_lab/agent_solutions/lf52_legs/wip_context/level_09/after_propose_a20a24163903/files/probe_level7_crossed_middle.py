"""Focused continuation from the verified crossed-role level-7 prefix."""

from collections import deque
import json

import gkm_try

from legs import _movable_bridge_board
from perception import arr, connected_components, safe_step


LEVEL_START = 331
CROSS_PREFIX = (
    3, 3, 1, 1, 3, 3, 3, (6, 7, 13), (6, 7, 25),
    4, 4, 4, 2, 2, 4, 4, 4, 2, (6, 43, 43), (6, 43, 55),
    1, 3, 3, 3, 1, 1, 4, 4, 4, (6, 43, 13), (6, 43, 25),
    3, 3, 3, 2, 2, 3, 3, 2, (6, 13, 43), (6, 13, 55),
    (6, 13, 55), (6, 25, 55), (6, 25, 55), (6, 37, 55),
    (6, 37, 55), (6, 49, 55), (6, 43, 55), (6, 55, 55),
    (6, 5, 55), (6, 17, 55),
    2, 3, 2, 2, 4, 4, 2,
    (6, 11, 55), (6, 23, 55), (6, 17, 55), (6, 29, 55),
)


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    _, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return tuple(sorted(carriers)), tuple(sorted(bridges)), tuple(sorted(pegs))


def legal(node, desired):
    source, destination = desired
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
        if legal(node, desired):
            return path, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            child_key = frame_key(child)
            if child_key in seen:
                continue
            seen.add(child_key); queue.append((child, path + (action,)))
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
    for action in CROSS_PREFIX:
        safe_step(env, action)
    print("L7_CROSS_ROOT", len(CROSS_PREFIX), compact(env), flush=True)

    continuation = []
    desired_moves = (
        ((24, 34), (24, 46)),
        ((54, 28), (42, 28)),
        ((18, 46), (30, 46)),
    )
    trace = []
    for desired in desired_moves:
        keys, searched = align(env, desired)
        trace.append((desired, None if keys is None else len(keys), searched, compact(env)))
        print("L7_CROSS_MIDDLE", trace[-1], flush=True)
        if keys is None:
            break
        for action in keys:
            safe_step(env, action); continuation.append(action)
        apply_move(env, desired, continuation)
    print("L7_CROSS_MIDDLE_RESULT", len(continuation), trace, compact(env))
    print("L7_CROSS_MIDDLE_ACTIONS", continuation)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
