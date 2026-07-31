"""Bounded symbolic search over level 3's selectable movable groups."""
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import MIRRORED_PAIR_ASCENT, MIRRORED_PAIR_MAZE_REUNION
from perception import connected_components


def reach_level_3(env):
    for action in MIRRORED_PAIR_ASCENT + MIRRORED_PAIR_MAZE_REUNION:
        env.step(action)


def pieces(frame):
    return connected_components(frame, colors=(1, 9, 10, 11), min_area=2)


def key(env):
    f = np.asarray(env.frame())
    # Preserve positions, shapes, and active/inactive status; discard timer pixels.
    return np.packbits(np.isin(f, (1, 9, 10, 11))).tobytes() + bytes(
        [1 if np.any(f == 10) else 2]
    )


def separation(frame):
    blobs = pieces(frame)
    boxes = [b.bbox for b in blobs]
    total = 0
    for i, a in enumerate(boxes):
        best = 1000
        for j, b in enumerate(boxes):
            if i == j:
                continue
            dr = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
            dc = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
            best = min(best, dr + dc)
        total += best
    return total


def inspect(env):
    reach_level_3(env)
    base = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    best = (separation(env.frame()), ())
    solution = None
    limit = 30000
    while queue and len(seen) < limit:
        node, path = queue.popleft()
        if len(path) >= 90:
            continue
        f = np.asarray(node.frame())
        choices = [1, 2, 3, 4]
        # One click per inactive group. Large color-1 blocks are one joint group.
        inactive = connected_components(f, colors=(1, 9), min_area=2)
        large_added = False
        for blob in inactive:
            if blob.color == 1:
                if large_added:
                    continue
                large_added = True
            r0, c0, r1, c1 = blob.bbox
            choices.append((6, (c0 + c1) // 2, (r0 + r1) // 2))
        for choice in choices:
            child = node.clone()
            if isinstance(choice, tuple):
                child.step(*choice)
                recorded = choice
            else:
                child.step(choice)
                recorded = choice
            new_path = path + (recorded,)
            if child.levels_completed > base:
                solution = new_path
                queue.clear()
                break
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            score = separation(child.frame())
            if score < best[0]:
                best = (score, new_path)
                print("best", score, "depth", len(new_path), "seen", len(seen),
                      "pieces", [(b.color, b.bbox) for b in pieces(child.frame())])
            queue.append((child, new_path))
        if len(seen) and len(seen) % 5000 == 0:
            print("progress", len(seen), "queue", len(queue), "depth", len(path))
    print("search", "seen", len(seen), "best", best[0],
          "solution", solution)


if __name__ == "__main__":
    A.run_program("m0r0", inspect)
