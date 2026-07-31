import numpy as np

import gkm_try as harness


def repeat(node, action, count):
    for _ in range(count):
        node.step(action)


def scan(start, selection, dy_range, dx_range):
    chosen = start.clone()
    chosen.step(*selection)
    for turns in range(4):
        oriented = chosen.clone()
        repeat(oriented, 5, turns)
        hits = []
        for dy in dy_range:
            row = oriented.clone()
            repeat(row, 1 if dy < 0 else 2, abs(dy))
            for dx in dx_range:
                node = row.clone()
                repeat(node, 3 if dx < 0 else 4, abs(dx))
                frame = np.asarray(node.frame())
                c3 = int(np.count_nonzero(frame[1:] == 3))
                if c3 or node.levels_completed > 4:
                    occupied = int(np.count_nonzero(frame[1:] != 15))
                    hits.append((int(node.levels_completed), c3, -occupied, dy, dx))
        hits.sort(reverse=True)
        print("ROT", turns, "TOP", hits[:25], "COUNT", len(hits))


def probe(env):
    harness.resumed_solve(env)
    repeat(env, 1, 3)
    repeat(env, 4, 8)
    env.step(6, 5, 38)
    repeat(env, 1, 11)
    repeat(env, 4, 15)
    print("LEFT_STAGED")
    scan(env, (6, 47, 47), range(-15, 4), range(-12, 7))


harness.A.run_program("cn04", probe)
