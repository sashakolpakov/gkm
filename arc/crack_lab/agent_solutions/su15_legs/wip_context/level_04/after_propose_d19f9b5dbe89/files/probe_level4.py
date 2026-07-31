import gkm_try as H
from perception import color_counts, connected_components, frame_delta


def compact(frame):
    blobs = connected_components(frame, min_area=4)
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in blobs
        if b.area < 3000
    ]


def probe(env):
    H.m.solve(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    base = env.frame()
    print("COLORS", color_counts(base))
    print("BLOBS", compact(base))
    a = __import__("numpy").asarray(base)
    for y in range(10, 63):
        runs = []
        start = 0
        for x in range(1, 65):
            if x == 64 or a[y, x] != a[y, start]:
                if int(a[y, start]) != 5:
                    runs.append((start, x - 1, int(a[y, start])))
                start = x
        if runs:
            print("ROW", y, runs)
    candidates = [
        b for b in connected_components(base, min_area=4)
        if b.area < 3000 and b.bbox[0] >= 10
    ]
    tests = [(54 + dx, 21 + dy) for dy in (-4, 0, 4) for dx in (-4, 0, 4)
             if dx or dy]
    tests += [(5, 57), (36, 29), (31, 27), (8, 41)]
    for x, y in tests:
        clone = env.clone()
        clone.step(6, x, y)
        ca = __import__("numpy").asarray(clone.frame())
        ys, xs = __import__("numpy").where(ca == 7)
        print("DIR", (x, y),
              None if not len(ys) else (round(float(ys.mean()), 1),
                                       round(float(xs.mean()), 1)),
              "N10", int((ca == 10).sum()), "LEVEL", clone.levels_completed)
    clone = env.clone()
    for i in range(10):
        ca = __import__("numpy").asarray(clone.frame())
        ys, xs = __import__("numpy").where(ca == 7)
        print("REL", i,
              None if not len(ys) else (round(float(ys.mean()), 1),
                                       round(float(xs.mean()), 1)),
              "N10", int((ca == 10).sum()), "LEVEL", clone.levels_completed)
        if clone.levels_completed != env.levels_completed or not len(ys):
            break
        try:
            clone.step(6, round(float(xs.mean())) - 4,
                       round(float(ys.mean())) + 4)
        except Exception as exc:
            print("REL_ERR", type(exc).__name__, str(exc))
            break
    clone = env.clone()
    for i, (dx, dy) in enumerate([(-4, 0)] * 4 + [(0, 4)] * 3):
        ca = __import__("numpy").asarray(clone.frame())
        ys, xs = __import__("numpy").where(ca == 7)
        print("HV", i, (round(float(ys.mean()), 1), round(float(xs.mean()), 1)),
              "N10", int((ca == 10).sum()), "LEVEL", clone.levels_completed)
        try:
            clone.step(6, round(float(xs.mean())) + dx,
                       round(float(ys.mean())) + dy)
        except Exception as exc:
            print("HV_ERR", type(exc).__name__, str(exc),
                  "LEVEL", clone.levels_completed)
            break
    print("HV_END", clone.levels_completed, clone.terminal())
    clone = env.clone()
    remove = [(5, 26), (11, 26), (31, 27), (36, 29),
              (8, 41), (33, 47), (30, 51)]
    for i, (x, y) in enumerate(remove):
        ca = __import__("numpy").asarray(clone.frame())
        ys, xs = __import__("numpy").where(ca == 7)
        print("STAGE", i, (round(float(ys.mean()), 1), round(float(xs.mean()), 1)),
              "N10", int((ca == 10).sum()), "LEVEL", clone.levels_completed)
        clone.step(6, x, y)
    ca = __import__("numpy").asarray(clone.frame())
    ys, xs = __import__("numpy").where(ca == 7)
    print("STAGE_END", (round(float(ys.mean()), 1), round(float(xs.mean()), 1)),
          "N10", int((ca == 10).sum()), "LEVEL", clone.levels_completed)
    for b in candidates:
        x, y = round(b.centroid[1]), round(b.centroid[0])
        for action in env.actions:
            clone = env.clone()
            try:
                clone.step(int(action), x, y)
                d = frame_delta(base, clone.frame())
                print("AT", (b.color, b.bbox), "ACT", action,
                      "DELTA", (d["count"], d["bbox"]),
                      "SAMPLES", d["samples"][:24],
                      "LEVEL", clone.levels_completed)
            except Exception as exc:
                print("AT", (b.color, b.bbox), "ACT", action,
                      "ERR", type(exc).__name__, str(exc)[:80])
    clone = env.clone()
    for i in range(36):
        ca = __import__("numpy").asarray(clone.frame())
        ys, xs = __import__("numpy").where(ca == 7)
        ten = connected_components(clone.frame(), colors=(10,), min_area=1)
        print("GREEDY", i,
              None if not len(ys) else (round(float(ys.mean()), 1),
                                       round(float(xs.mean()), 1)),
              "N7", len(ys),
              "TEN", [b.bbox for b in ten if b.bbox[0] >= 10],
              "LEVEL", clone.levels_completed)
        if clone.levels_completed != env.levels_completed or not len(ys):
            break
        try:
            clone.step(6, 5, 57)
        except Exception as exc:
            print("GREEDY_ERR", i, type(exc).__name__, str(exc),
                  "LEVEL", clone.levels_completed, "TERM", clone.terminal())
            ca = __import__("numpy").asarray(clone.frame())
            ys, xs = __import__("numpy").where(ca == 7)
            print("AFTER_ERR", len(ys),
                  None if not len(ys) else (float(ys.mean()), float(xs.mean())))
            break


levels, path, err = H.A.run_program("su15", probe)
print("DONE", levels, len(path), err)
