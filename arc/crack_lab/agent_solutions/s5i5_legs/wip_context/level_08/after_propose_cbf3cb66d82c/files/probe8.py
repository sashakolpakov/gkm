import sys
import heapq

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def probe(env):
    for level in range(1, 8):
        getattr(players, f"play_level_{level}")(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    controls = [(39, 4), (45, 4), (53, 4), (59, 4),
                (39, 11), (45, 11), (53, 11), (59, 11),
                (49, 18), (8, 56), (14, 56), (22, 56), (28, 56)]
    def marker_row(node):
        f = arr(node.frame())
        rows = [int(r) for r, c in zip(*((f == 13).nonzero())) if r < 40]
        return min(rows) if rows else 43

    counter = 0
    heap = [(-marker_row(env), 0, counter, env.clone(), [])]
    seen = {arr(env.frame()).tobytes()}
    best = marker_row(env)
    while heap and len(seen) < 12000:
        _, depth, _, node, path = heapq.heappop(heap)
        for xy in controls:
            child = node.clone()
            child.step(6, *xy)
            key = arr(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_path = path + [xy]
            row = marker_row(child)
            if row > best:
                best = row
                print("BEST", best, "DEPTH", len(child_path),
                      "SEEN", len(seen), child_path, flush=True)
            if child.levels_completed > 7:
                print("FOUND", len(child_path), child_path)
                return
            counter += 1
            heapq.heappush(
                heap, (-row, depth + 1, counter, child, child_path))
    print("NO_PATH", len(seen), "BEST", best)


levels, path, err = A.run_program("s5i5", probe)
print("DONE", levels, len(path), err)
