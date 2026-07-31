import numpy as np
import gkm_try as H


def pts(frame, color):
    a = np.asarray(frame)
    return {(int(y), int(x)) for y, x in zip(*np.where(a == color))}


def probe(env):
    H.m.solve(env)
    base = env.frame()
    sevens = pts(base, 7)
    tens = {(y, x) for y, x in pts(base, 10) if y >= 10}
    print("BASE7", sorted(sevens), "BASE10", sorted(tens))
    for y, x in sorted(tens):
        clone = env.clone()
        clone.step(6, x, y)
        after7, after10 = pts(clone.frame(), 7), pts(clone.frame(), 10)
        print("CLICK", (y, x),
              "TEN_GONE", sorted(pts(base, 10) - after10),
              "TEN_NEW", sorted(after10 - pts(base, 10)),
              "D7", (round(sum(y for y, _ in after7) / len(after7), 1),
                     round(sum(x for _, x in after7) / len(after7), 1)))


levels, path, err = H.A.run_program("su15", probe)
print("DONE", levels, len(path), err)
