import heapq
import itertools
import numpy as np

import gkm_try as H


def state_info(env):
    a = np.asarray(env.frame())
    ys, xs = np.where(a == 7)
    if not len(ys):
        return None
    row, col = float(ys.mean()), float(xs.mean())
    play_ten = tuple((int(y), int(x)) for y, x in zip(*np.where(a[10:63] == 10)))
    play_ten = tuple((y + 10, x) for y, x in play_ten)
    return row, col, play_ten


def probe(env):
    H.m.solve(env)
    base = env.levels_completed
    initial = state_info(env)
    fixed = [(x, y) for y, x in initial[2]]
    ring = (5, 57)
    serial = itertools.count()
    heap = [(0, 0, next(serial), env.clone(), [])]
    seen = {np.asarray(env.frame()).tobytes()}
    expanded = 0
    while heap and expanded < 2400:
        _, depth, _, node, path = heapq.heappop(heap)
        expanded += 1
        info = state_info(node)
        if info is None or depth >= 22:
            continue
        row, col, remaining = info
        dynamic = [
            (round(col) - 4, round(row)),
            (round(col) + 4, round(row)),
            (round(col), round(row) - 4),
            (round(col), round(row) + 4),
        ]
        for x, y in fixed + [ring] + dynamic:
            child = node.clone()
            try:
                child.step(6, int(x), int(y))
            except Exception:
                continue
            child_path = path + [(6, int(x), int(y))]
            if child.levels_completed > base:
                print("FOUND", child_path, "EXPANDED", expanded)
                return
            key = np.asarray(child.frame()).tobytes()
            if key in seen or child.terminal():
                continue
            seen.add(key)
            ci = state_info(child)
            if ci is None:
                continue
            cr, cc, rem = ci
            distance = abs(cr - 57) + abs(cc - 5)
            score = distance + len(rem) * 2 + len(child_path) * 0.1
            heapq.heappush(
                heap, (score, len(child_path), next(serial), child, child_path)
            )
    print("NO_PATH", "EXPANDED", expanded, "SEEN", len(seen))


levels, path, err = H.A.run_program("su15", probe)
print("DONE", levels, len(path), err)
