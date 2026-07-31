import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr
from players import play_level_1, play_level_2, play_level_3


CONTROLS = [
    ("orange_down", (6, 48)),
    ("orange_up", (12, 48)),
    ("magenta_down", (51, 48)),
    ("magenta_up", (57, 48)),
    ("contract_all", (28, 54)),
    ("expand_all", (34, 54)),
    ("brown_down", (6, 57)),
    ("brown_up", (12, 57)),
    ("blue_down", (51, 57)),
    ("blue_up", (57, 57)),
]
TARGET = {(9, 31), (10, 30), (10, 32), (11, 31)}


def marker_row(frame):
    grid = arr(frame)
    moving = [
        int(r) for r, c in zip(*((grid[:45] == 13).nonzero()))
        if (int(r), int(c)) not in TARGET
    ]
    return min(moving) if moving else 10


def probe(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)
    serial = 0
    start_key = arr(env.frame())[:45].tobytes()
    queue = [(marker_row(env.frame()), 0, serial, env, [])]
    seen = {start_key}
    best = marker_row(env.frame())
    steps = 0
    while queue and len(seen) < 6000:
        _, depth, _, node, path = heapq.heappop(queue)
        if depth >= 70:
            continue
        for name, (x, y) in CONTROLS:
            child = node.clone()
            child.step(6, x, y)
            steps += 1
            key = arr(child.frame())[:45].tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_path = path + [name]
            if child.levels_completed > 3:
                print("WIN", len(child_path), child_path)
                return
            row = marker_row(child.frame())
            if row < best:
                best = row
                print("BEST", best, "depth", len(child_path), child_path)
            serial += 1
            # Prefer upward marker progress, then shorter paths.
            heapq.heappush(
                queue, (row + 0.02 * len(child_path), len(child_path),
                        serial, child, child_path)
            )
    print("NO_WIN", "seen", len(seen), "steps", steps, "best", best,
          "queue", len(queue))


arena.run_program("s5i5", probe)
