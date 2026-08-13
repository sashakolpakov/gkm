"""Greedily minimize key alignment paths around level-7's verified macros."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board
from perception import arr, safe_step


PREFIX_MOVES = 331
KEYS = (1, 2, 3, 4)
KNOWN_ALIGNMENTS = (
    (3, 3, 1, 1, 3, 3, 3),
    (4, 4, 4, 2, 2, 3, 3, 2),
    (1, 4, 4, 1, 1, 4, 4, 4),
    (3, 3, 3, 2, 2, 4, 4, 4, 2),
    (), (), (), (), (),
    (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2),
    (),
    (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4),
    (3, 3, 3, 2, 2, 4, 4, 2),
    (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2),
    (), (), (), (), (), (),
    (2, 3, 3, 3, 2, 2),
    (1, 1, 4, 4, 4, 2, 2, 4, 2),
    (1, 3, 1, 3, 3, 2, 4, 2, 2),
)
MACROS = (
    ((12, 6), (24, 6)),
    ((42, 12), (54, 12)),
    ((12, 42), (24, 42)),
    ((42, 42), (54, 42)),
    ((54, 12), (54, 24)),
    ((54, 24), (54, 36)),
    ((54, 36), (54, 48)),
    ((54, 42), (54, 54)),
    ((54, 4), (54, 16)),
    ((54, 10), (54, 22)),
    ((54, 16), (54, 28)),
    ((24, 34), (24, 46)),
    ((54, 28), (42, 28)),
    ((18, 46), (30, 46)),
    ((24, 46), (36, 46)),
    ((30, 46), (42, 46)),
    ((36, 46), (36, 58)),
    ((36, 18), (36, 30)),
    ((42, 6), (42, 18)),
    ((42, 18), (42, 30)),
    ((42, 30), (30, 30)),
    ((24, 54), (36, 54)),
    ((42, 54), (30, 54)),
)


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def clicks(move):
    source, destination = move
    return (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    )


def macro_child(node, move, base_level):
    before_frame = arr(node.frame())[1:, :].tobytes()
    _, _, before_bridges, before_pegs = _movable_bridge_board(node.frame())
    source, destination = move
    kind = "peg" if source in before_pegs else (
        "bridge" if source in before_bridges else None
    )
    child = node.clone()
    for action in clicks(move):
        safe_step(child, action)
    if child.levels_completed > base_level:
        return child
    after_frame = arr(child.frame())
    if (
        after_frame[1:, :].tobytes() != before_frame
        and not ((after_frame == 2) | (after_frame == 3)).any()
    ):
        return child
    if kind is None:
        return None
    _, _, after_bridges, after_pegs = _movable_bridge_board(child.frame())
    after_pieces = after_pegs if kind == "peg" else after_bridges
    if destination not in after_pieces or source in after_pieces:
        return None
    return child


def shortest_alignment(root, move, base_level,
                       max_states=2000, max_depth=20):
    queue = deque([()])
    seen = {key(root)}
    while queue and len(seen) <= max_states:
        path = queue.popleft()
        node = root.clone()
        for action in path:
            safe_step(node, action)
        if macro_child(node, move, base_level) is not None:
            return path
        if len(path) >= max_depth:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append(path + (action,))
    return None


def observe(env):
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    for action in admitted[:PREFIX_MOVES]:
        env.step(action)
    print("ENTRY", {"level": env.levels_completed, "prefix": PREFIX_MOVES})

    node = env.clone()
    base_level = int(node.levels_completed)
    candidate = []
    for index, move in enumerate(MACROS, 1):
        known = KNOWN_ALIGNMENTS[index - 1]
        if index <= 9 or not known:
            alignment = known
        else:
            alignment = shortest_alignment(
                node, move, base_level,
                max_depth=max(0, len(known) - 1),
            )
            if alignment is None:
                alignment = known
        if alignment is None:
            print("FAILED", {"macro": index, "move": move,
                             "candidate_len": len(candidate)})
            return
        for action in alignment:
            safe_step(node, action)
            candidate.append(action)
        for action in clicks(move):
            safe_step(node, action)
            candidate.append(action)
        print("ALIGN", {"macro": index, "keys": alignment,
                        "candidate_len": len(candidate),
                        "level": node.levels_completed})

    print("CANDIDATE", {"level": node.levels_completed,
                        "actions": len(candidate),
                        "keys": len(candidate) - 2 * len(MACROS),
                        "path": candidate})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
