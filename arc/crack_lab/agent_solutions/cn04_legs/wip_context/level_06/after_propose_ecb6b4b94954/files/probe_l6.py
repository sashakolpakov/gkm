import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from perception import action_deltas, color_counts, object_candidates


def probe(env):
    while env.levels_completed < 5 and not env.terminal():
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)

    print("level", env.levels_completed + 1, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    objects = object_candidates(env.frame())
    print("objects")
    for obj in objects:
        if obj["color"] != 9:
            print(obj)
    frame = env.frame()
    symbols = {4: "#", 8: "A", 9: ".", 14: "o"}
    print("crop")
    for row in range(0, 31):
        line = "".join(symbols[int(frame[row, col])] for col in range(0, 31))
        if line.strip("."):
            print(f"{row:02}", line)
    print("key_deltas")
    for action, delta in action_deltas(env).items():
        print(action, {k: delta[k] for k in ("count", "bbox")})

    print("select_deltas")
    for obj in objects:
        if obj["color"] in (8, 10):
            row, col = (round(v) for v in obj["centroid"])
            clone = env.clone()
            before = clone.frame().copy()
            clone.step(6, col, row)
            from perception import frame_delta
            delta = frame_delta(before, clone.frame())
            print((col, row), "level", clone.levels_completed,
                  {k: delta[k] for k in ("count", "bbox")})
            for turns in range(4):
                turned = clone.clone()
                for _ in range(turns):
                    turned.step(5)
                f = turned.frame()
                cells = {}
                for rr in range(0, 64, 3):
                    for cc in range(0, 64, 3):
                        value = int(f[rr, cc])
                        if value not in (4, 9):
                            cells[(rr // 3, cc // 3)] = value
                print(" ", "turn", turns, "colors", color_counts(f),
                      "cells", sorted(cells.items()))

    targets = {}
    for rr in range(0, 64, 3):
        for cc in range(0, 64, 3):
            if int(frame[rr, cc]) == 4:
                targets[(rr // 3, cc // 3)] = 4
    print("target_cells", sorted(targets))


arena.run_program("cn04", probe)
