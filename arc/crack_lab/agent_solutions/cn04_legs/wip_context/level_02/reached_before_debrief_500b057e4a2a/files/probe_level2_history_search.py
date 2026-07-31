"""Best-first search over pose plus ordered rotation/socket engagements."""
import sys
import heapq
from itertools import count

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


EDGES = ((1, 3), (0, 2), (0, 5), (2, 4), (6, 7), (6, 8), (7, 11), (9, 10))


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    serial = count()
    heap = [(0, 0, next(serial), root.clone(), [], ())]
    seen = {(avatar_key(root), ())}
    max_progress = 0
    while heap and len(seen) < 45000:
        _, depth, _, node, path, history = heapq.heappop(heap)
        if depth >= 92 or node.terminal():
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > 1:
                print("solved", len(child_path), child_path, "history", history)
                return
            child_history = history
            if action == 5:
                events = tuple(
                    i for i, edge in enumerate(EDGES)
                    if all(covered(child, pegs[p]) for p in edge)
                )
                for event in events:
                    if event not in child_history:
                        child_history += (event,)
            key = (avatar_key(child), child_history)
            if key in seen:
                continue
            seen.add(key)
            progress = len(child_history)
            if progress > max_progress:
                max_progress = progress
                print("progress", progress, "depth", depth + 1, "history", child_history)
            heapq.heappush(
                heap,
                (-progress, depth + 1, next(serial), child, child_path, child_history),
            )
    print("unsolved", "states", len(seen), "frontier", len(heap), "max_progress", max_progress)


arena.run_program("cn04", probe)
