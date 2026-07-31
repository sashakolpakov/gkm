import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import action_deltas, block_signatures, color_counts, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def probe(env):
    solver.solve(env)
    frame = env.frame()
    print("LEVEL", int(env.levels_completed), "COLORS", color_counts(frame))
    objects = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color != 3
    ]
    print("OBJECTS", objects)
    print("AVATAR", avatar(frame))
    print("MOVE_DELTAS", {k: (v["count"], v["bbox"]) for k, v in action_deltas(env, (1, 2, 3, 4)).items()})
    for point in ((51, 18), (52, 19), (46, 28), (51, 27), (51, 36), (51, 45)):
        clone = env.clone()
        clone.step(6, *point)
        delta = frame_delta(frame, clone.frame())
        print("CLICK", point, "DELTA", delta["count"], delta["bbox"], "COLORS", color_counts(clone.frame()))
        changed = {
            k: v for k, v in block_signatures(clone.frame(), 2).items()
            if k[0] < 28 and k[1] < 19 and v != block_signatures(frame, 2)[k]
        }
        print("TILE_CHANGES", point, changed)

    # Discover coordinate affordances by grouping distinct gameplay outcomes.
    outcomes = {}
    base_play = np.asarray(frame)[8:56, :38].tobytes()
    for y in range(16, 48, 2):
        for x in range(42, 62, 2):
            clone = env.clone()
            clone.step(6, x, y)
            play = np.asarray(clone.frame())[8:56, :38].tobytes()
            if play != base_play:
                outcomes.setdefault(play, []).append((x, y))
    print("CLICK_OUTCOMES", [points for points in outcomes.values()])

    # Compact 4x4 occupancy map of the playable left side.
    arr = np.asarray(frame)
    symbols = {2: ".", 3: " ", 4: "#", 5: "5", 8: "8", 9: "9", 11: "G", 13: "D", 14: "A"}
    for r in range(0, 56, 4):
        row = []
        for c in range(0, 40, 4):
            vals, counts = np.unique(arr[r:r + 4, c:c + 4], return_counts=True)
            color = int(vals[int(np.argmax(counts))])
            row.append(symbols.get(color, "?"))
        print(f"MAP{r // 4:02d}", "".join(row))

    # Exact logical tiles: gameplay objects and corridors align to 2x2 cells.
    exact = {2: ".", 3: " ", 4: "#", 5: "5", 6: "6", 7: "7", 8: "8",
             9: "9", 11: "G", 12: "C", 13: "D", 14: "A"}
    for r in range(8, 56, 2):
        row = []
        for c in range(0, 38, 2):
            vals = np.unique(arr[r:r + 2, c:c + 2])
            row.append(exact.get(int(vals[0]), "?") if len(vals) == 1 else "*")
        print(f"T{r // 2:02d}", "".join(row))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
