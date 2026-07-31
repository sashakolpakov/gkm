import numpy as np
import gkm_try as H


def center(env):
    a = np.asarray(env.frame())
    ys, xs = np.where(a == 7)
    return float(ys.mean()), float(xs.mean()), int((a[10:63] == 10).sum())


def probe(env):
    H.m.solve(env)
    base = env.levels_completed
    for name, count in (("L", 4), ("D", 2)):
        for _ in range(count):
            r, c, n = center(env)
            print(name, (round(r, 1), round(c, 1)), n, env.levels_completed)
            env.step(6, round(c) - (4 if name == "L" else 0),
                     round(r) + (4 if name == "D" else 0))
    for i in range(10):
        r, c, n = center(env)
        print("RING", i, (round(r, 1), round(c, 1)), n, env.levels_completed)
        if env.levels_completed > base:
            return
        env.step(6, 5, 57)


levels, path, err = H.A.run_program("su15", probe)
print("DONE", levels, len(path), err)
