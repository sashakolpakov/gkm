import numpy as np

import gkm_try as harness
import players


def centers(frame):
    pts = np.argwhere(np.asarray(frame) == 8)
    return sorted((int(r), int(c)) for r, c in pts
                  if int(frame[r - 1:r + 2, c - 1:c + 2].sum()) == 72)


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    specs = [
        ("base", None, 0, 0, 0),
        ("a", (6, 36, 21), 1, 2, -7),
        ("b", (6, 45, 42), 3, -4, -12),
        ("c", (6, 18, 48), 1, -5, -3),
    ]
    sets = {}
    for name, select, turns, dy, dx in specs:
        node = env.clone()
        if select:
            node.step(*select)
        for _ in range(turns):
            node.step(5)
        raw = centers(node.frame())
        shifted = [(r + 3 * dy, c + 3 * dx) for r, c in raw]
        sets[name] = set(shifted)
        print(name, "raw", raw, "placed", shifted)
    for a, aset in sets.items():
        for b, bset in sets.items():
            if a < b:
                print("MATCH", a, b, sorted(aset & bset))


harness.A.run_program("cn04", probe)
