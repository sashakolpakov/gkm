"""Coordinate-only frontier of the crossed level-7 final board."""

from collections import deque
import json

import gkm_try

from perception import arr, safe_step
from probe_level7_crossed_middle import CROSS_PREFIX
from probe_level7_crossed_keys import compact, legal_moves
from probe_level7_crossed_follow import (
    BRIDGE_LOAD,
    BRIDGE_LOAD_ALIGN,
    PEG_TRANSPORT,
    PEG_UNLOAD,
)
from probe_level7_crossed_finalboard import (
    BRIDGE_TRANSPORT,
    BRIDGE_UNLOAD,
    CROSSED_EXIT,
)


LEVEL_START = 331
TO_FINAL = (
    CROSS_PREFIX + PEG_TRANSPORT + PEG_UNLOAD
    + BRIDGE_LOAD_ALIGN + BRIDGE_LOAD
    + BRIDGE_TRANSPORT + BRIDGE_UNLOAD + CROSSED_EXIT
)


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def apply_move(node, move):
    _, source, destination = move
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    for action in TO_FINAL:
        safe_step(env, action)

    queue = deque([(env.clone(), ())]); seen = {frame_key(env)}; rows = []
    while queue and len(seen) <= 200:
        node, path = queue.popleft()
        moves = legal_moves(node)
        rows.append((path, compact(node), moves, int(node.levels_completed)))
        if len(path) >= 6 or int(node.levels_completed) >= 7:
            continue
        for move in moves:
            child = node.clone(); apply_move(child, move)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (move,)))
    print("L7_CROSS_FINAL_FRONTIER", len(seen), len(rows))
    for row in rows:
        print("L7_CROSS_FINAL_STATE", row)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
