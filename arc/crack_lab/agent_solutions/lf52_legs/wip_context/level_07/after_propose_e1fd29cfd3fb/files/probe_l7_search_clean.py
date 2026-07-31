import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import connected_components


def board(frame):
    return tuple(frozenset(part) for part in _movable_bridge_board(frame))


def state_key(frame):
    geometry = board(frame)
    agents = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 7, 11))
        if blob.area < 100
    )
    return geometry, agents


def macros(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    out = [(action,) for action in (1, 2, 3, 4)]
    for source in sorted(bridges | pegs):
        for destination in sorted(slots | carriers):
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if ((dr == 0) != (dc == 0)) and abs(dr + dc) == 12:
                out.append((source, destination))
    return out


def apply_macro(node, macro):
    if len(macro) == 1:
        node.step(macro[0])
    else:
        play_lattice_moves(node, (macro,))


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(*action) if isinstance(action, list) else env.step(action)

    play_actions(env, (3, 3, 1, 1, 3, 3, 3))
    play_lattice_moves(env, (((12, 6), (24, 6)),))
    play_actions(env, (4, 4, 4, 2, 2, 4, 4, 4, 2))
    play_lattice_moves(env, (((42, 42), (54, 42)),))
    play_actions(env, (1, 3, 3, 3, 1, 1, 4, 4, 4))
    play_lattice_moves(env, (((12, 42), (24, 42)),))
    play_actions(env, (3, 3, 3, 2, 2, 3, 3, 2))
    play_lattice_moves(env, (((42, 12), (54, 12)),))

    play_lattice_moves(env, (
        ((54, 12), (54, 24)),
        ((54, 24), (54, 36)),
        ((54, 36), (54, 48)),
        ((54, 42), (54, 54)),
        ((54, 4), (54, 16)),
    ))
    play_actions(env, (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2))
    play_lattice_moves(env, (
        ((54, 10), (54, 22)),
        ((54, 16), (54, 28)),
    ))
    play_actions(env, (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4))
    play_lattice_moves(env, (((24, 34), (24, 46)),))
    play_actions(env, (3, 3, 3, 2, 2, 4, 4, 2))
    play_lattice_moves(env, (((54, 28), (42, 28)),))
    play_actions(env, (1, 3, 3, 1, 1, 4, 4, 4, 3, 1, 1, 4, 4, 4, 2))
    play_lattice_moves(env, (
        ((18, 46), (30, 46)),
        ((24, 46), (36, 46)),
        ((30, 46), (42, 46)),
        ((36, 46), (36, 58)),
    ))
    base_level = env.levels_completed
    root = env.clone()
    queue = deque([(root, ())])
    seen = {state_key(root.frame())}
    found = None
    while queue and len(seen) < 300:
        node, path = queue.popleft()
        if len(path) >= 40:
            continue
        for macro in macros(node.frame()):
            child = node.clone()
            apply_macro(child, macro)
            child_path = path + (macro,)
            if child.levels_completed > base_level:
                found = child_path
                queue.clear()
                break
            key = state_key(child.frame())
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("SEARCH", len(seen), "FOUND", found)


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
