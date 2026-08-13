"""Deterministic key probes from the verified crossed-role middle state."""

import json

import gkm_try

from legs import _movable_bridge_board
from perception import arr, connected_components, safe_step

from probe_level7_crossed_middle import CROSS_PREFIX


LEVEL_START = 331


def compact(node):
    _, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return tuple(sorted(carriers)), tuple(sorted(bridges)), tuple(sorted(pegs))


def legal_moves(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    occupied = bridges | pegs
    result = []
    for kind, sources in (("B", bridges), ("P", pegs)):
        for source in sorted(sources):
            for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                destination = source[0] + dr, source[1] + dc
                midpoint = source[0] + dr // 2, source[1] + dc // 2
                if (
                    destination in slots | carriers
                    and destination not in occupied
                    and midpoint in occupied | fixed
                ):
                    result.append((kind, source, destination))
    return tuple(result)


def variants(path):
    inverse = {1: 2, 2: 1, 3: 4, 4: 3}
    horizontal = {1: 1, 2: 2, 3: 4, 4: 3}
    vertical = {1: 2, 2: 1, 3: 3, 4: 4}
    return {
        path,
        tuple(reversed(path)),
        tuple(inverse[action] for action in reversed(path)),
        tuple(horizontal[action] for action in path),
        tuple(vertical[action] for action in path),
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    for action in CROSS_PREFIX:
        safe_step(env, action)
    root = env.clone()

    bases = (
        (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4),
        (3, 3, 3, 2, 2, 4, 4, 2),
        (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2),
        (2, 3, 2, 2, 4, 4, 2),
        (1, 4, 4, 1, 1, 4, 4, 4),
        (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2),
    )
    paths = set()
    for base in bases:
        paths.update(variants(base))
        for cut in range(1, len(base)):
            paths.add(base[:cut])
    for action in (1, 2, 3, 4):
        for count in range(1, 13):
            paths.add((action,) * count)

    observations = {}
    for path in sorted(paths, key=lambda value: (len(value), value)):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        key = arr(node.frame())[1:, :].tobytes()
        moves = legal_moves(node)
        if key not in observations or len(path) < len(observations[key][0]):
            observations[key] = path, compact(node), moves

    useful = sorted(
        (
            (len(path), path, state, moves)
            for path, state, moves in observations.values()
            if moves
        ),
        key=lambda item: (item[0], item[1]),
    )
    print("L7_CROSS_KEYS", len(paths), len(observations), len(useful))
    for item in useful:
        print("L7_CROSS_KEY_STATE", item)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
