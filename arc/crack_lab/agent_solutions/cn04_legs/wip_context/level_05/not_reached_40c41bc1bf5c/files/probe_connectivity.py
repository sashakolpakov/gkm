import numpy as np

import gkm_try as harness
from perception import bounded_bfs, replay


def groups(env):
    frame = np.asarray(env.frame())
    occupied = frame[1:] != 15
    seen = np.zeros_like(occupied, dtype=bool)
    count = 0
    rows, cols = occupied.shape
    for r in range(rows):
        for c in range(cols):
            if not occupied[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and occupied[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def occupied_area(env):
    return int(np.count_nonzero(np.asarray(env.frame())[1:] != 15))


def connect_one(node, limit, depth):
    start = groups(node)
    path = bounded_bfs(
        node,
        lambda e, p: e.levels_completed > 4 or groups(e) < start,
        max_states=limit,
        max_depth=depth,
    )
    print("CONNECT", start, path)
    return replay(node, path) if path is not None else node


def probe(env):
    harness.resumed_solve(env)
    print("START", groups(env))
    node = connect_one(env, 2500, 12)
    print("STAGE1", groups(node), node.levels_completed)
    node.step(6, 47, 47)
    node = connect_one(node, 6000, 22)
    print("STAGE2", groups(node), node.levels_completed)
    node.step(6, 5, 38)
    node = connect_one(node, 12000, 28)
    print("FINAL", groups(node), node.levels_completed, "AREA", occupied_area(node))


harness.A.run_program("cn04", probe)
