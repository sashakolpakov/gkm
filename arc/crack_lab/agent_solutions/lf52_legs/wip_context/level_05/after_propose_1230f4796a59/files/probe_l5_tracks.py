import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_state
from perception import arr, block_signatures, connected_components


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def physical(env):
    frame = arr(env.frame())
    return frame[1:, :].tobytes()


def compact(env):
    slots, pegs, carriers, bridges, borders, selected = _bridge_carrier_state(
        env.frame()
    )
    return (
        tuple(sorted(pegs)),
        tuple(sorted(carriers)),
        tuple(sorted(bridges)),
        tuple(sorted(borders)),
        len(slots),
        env.levels_completed,
    )


def show(label, env):
    print(label, compact(env))
    print("SPECIAL", [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(1, 7, 11, 12, 14, 15), min_area=2
        )
    ])
    symbols = {14: "P", 12: "C", 15: "B", 1: "o", 7: "g", 11: "c",
               9: "#", 5: "|"}
    signatures = block_signatures(env.frame(), cell=6)
    print("MAP")
    for row in range(11):
        print("".join(
            next((symbols[color] for color in symbols
                  if color in signatures[(row, col)]), ".")
            for col in range(11)
        ))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX:
        act(env, action)

    paths = {
        "ROOT": (),
        "UP3": (1, 1, 1),
        "UP3_RIGHT3": (1, 1, 1, 4, 4, 4),
        "LEFT3_DOWN3": (3, 3, 3, 2, 2, 2),
        "RIGHT3_DOWN3": (4, 4, 4, 2, 2, 2),
    }
    nodes = {}
    for label, path in paths.items():
        node = env.clone()
        for action in path:
            node.step(action)
        nodes[label] = node
        show(label, node)
        next_states = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            next_states.append((action, compact(child)))
        print("NEXT", next_states)

    base = env.levels_completed
    queue = deque([()])
    seen = {physical(env)}
    interesting = []
    solution = None
    while queue and len(seen) < 1200:
        path = queue.popleft()
        node = env.clone()
        for action in path:
            node.step(action)
        if len(path) >= 48:
            continue
        for action in (1, 2, 3, 4):
            child_path = path + (action,)
            child = node.clone()
            child.step(action)
            if child.levels_completed > base:
                solution = child_path
                queue.clear()
                break
            key = physical(child)
            if key in seen:
                continue
            seen.add(key)
            before_pegs = compact(node)[0]
            after_pegs = compact(child)[0]
            if len(after_pegs) != len(before_pegs):
                interesting.append((child_path, before_pegs, after_pegs,
                                    compact(child)))
            queue.append(child_path)
    print("BFS", len(seen), "QUEUE", len(queue), "SOLUTION", solution)
    print("PEG_CHANGES", interesting[:20])


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)
