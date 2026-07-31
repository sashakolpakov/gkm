"""Find and verify the large-pair reunion states on level 3."""
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import MIRRORED_PAIR_ASCENT, MIRRORED_PAIR_MAZE_REUNION
from perception import connected_components


def inspect(env):
    for action in MIRRORED_PAIR_ASCENT + MIRRORED_PAIR_MAZE_REUNION:
        env.step(action)
    base = env.levels_completed

    def avatar_blobs(node):
        return connected_components(node.frame(), colors=(10,), min_area=4)

    def avatar_key(node):
        return np.packbits(np.asarray(node.frame()) == 10).tobytes()

    queue = deque([(env.clone(), ())])
    seen = {avatar_key(env)}
    reunions = []
    while queue:
        node, path = queue.popleft()
        blobs = avatar_blobs(node)
        if len(blobs) == 1:
            reunions.append((len(path), path, blobs[0].bbox, node.levels_completed))
            if len(reunions) >= 8:
                break
        if len(path) >= 80:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            k = avatar_key(child)
            if k not in seen:
                seen.add(k)
                queue.append((child, path + (action,)))
    print("avatar_states", len(seen), "reunions", reunions)
    if reunions:
        test = env.clone()
        for action in reunions[0][1]:
            test.step(action)
        print("verified", test.levels_completed, [
            (b.color, b.bbox, b.area)
            for b in connected_components(test.frame(), colors=(9, 10), min_area=2)
        ])
        for action in (1, 2, 3, 4):
            child = test.clone()
            child.step(action)
            print("joined_move", action, [
                b.bbox for b in connected_components(child.frame(), colors=(10,), min_area=4)
            ], child.levels_completed)
    assert env.levels_completed == base


if __name__ == "__main__":
    A.run_program("m0r0", inspect)
