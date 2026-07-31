"""Compact clean-room observations for dc22 level 6."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


def observe(env):
    solve.solve(env)
    frame = perception.arr(env.frame())
    print("STATE", env.levels_completed, env.terminal(), frame.shape, env.actions)
    print("COLORS", perception.color_counts(frame))
    objects = [
        obj for obj in perception.object_candidates(frame, min_area=2)
        if obj["area"] < 1000
    ]
    for obj in objects:
        print(
            "OBJ",
            obj["color"],
            obj["bbox"],
            obj["area"],
            tuple(round(v, 1) for v in obj["centroid"]),
        )
    for action in (1, 2, 3, 4):
        clone = env.clone()
        before = frame.copy()
        clone.step(action)
        print(
            "ACT",
            action,
            perception.frame_delta(before, clone.frame()),
            "LEVEL",
            clone.levels_completed,
        )
    symbols = "0123456789ABCDEF"
    print("DOMINANT")
    for row in range(0, 64, 4):
        cells = []
        for col in range(0, 64, 4):
            block = frame[row:row + 4, col:col + 4]
            counts = perception.color_counts(block)
            cells.append(symbols[max(counts, key=counts.get)])
        print(f"{row // 4:02d}", "".join(cells))
    print("CLICKS")
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            clone = env.clone()
            clone.step(6, x, y)
            delta = perception.frame_delta(frame, clone.frame())
            meaningful = [
                sample for sample in delta["samples"]
                if sample[:2] != (63, 0)
            ]
            if meaningful:
                print("CLICK", (x, y), delta["count"], delta["bbox"], meaningful[:12])


arena.run_program("dc22", observe)
