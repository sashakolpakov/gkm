import numpy as np

import gkm_try as harness


def move(node, action, count):
    for _ in range(max(0, count)):
        node.step(action)


def probe(env):
    harness.resumed_solve(env)
    for turns in range(4):
        oriented = env.clone()
        move(oriented, 5, turns)
        base_colors = set(int(v) for v in np.unique(oriented.frame()[1:]))
        hits = []
        for dy in range(-8, 13):
            row = oriented.clone()
            move(row, 1 if dy < 0 else 2, abs(dy))
            for dx in range(-10, 13):
                node = row.clone()
                move(node, 3 if dx < 0 else 4, abs(dx))
                colors = set(int(v) for v in np.unique(node.frame()[1:]))
                novel = tuple(sorted(colors - base_colors))
                c3 = int(np.count_nonzero(np.asarray(node.frame())[1:] == 3))
                if novel or node.levels_completed > 4:
                    hits.append((c3, dy, dx, novel, int(node.levels_completed)))
        hits.sort(reverse=True)
        print("ROT", turns, "BASE", sorted(base_colors), "HITS", hits[:30],
              "COUNT", len(hits))


harness.A.run_program("cn04", probe)
