import importlib.util
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import ACTION_NAME, action_deltas, block_signatures, color_counts, connected_components
from legs import _avatar_pos, clone_after, fast_reach


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("local_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, colors=(5, 8, 9), min_area=4)
    ]


def transitions(before, after):
    a, b = np.asarray(before), np.asarray(after)
    return sorted((int(x), int(y), int(n)) for (x, y), n in Counter(zip(a.ravel(), b.ravel())).items() if x != y)


def coarse(frame):
    f = np.asarray(frame)
    names = {0: ".", 1: "1", 5: "#", 8: "G", 9: "A", 11: "b", 15: "f"}
    rows = []
    for r in range(2, 63, 6):
        row = []
        for c in range(2, 63, 6):
            vals, counts = np.unique(f[r:r+5, c:c+5], return_counts=True)
            row.append(names.get(int(vals[np.argmax(counts)]), "?"))
        rows.append("".join(row))
    return rows


def tiles(frame):
    f = np.asarray(frame)
    names = {0: ".", 1: "1", 5: "#", 8: "G", 9: "A", 11: "b", 15: "f"}
    rows = []
    mixed = []
    for i, r in enumerate(range(8, 62, 6)):
        row = []
        for j, c in enumerate(range(2, 62, 6)):
            vals, counts = np.unique(f[r:r+5, c:c+5], return_counts=True)
            pairs = tuple((int(v), int(n)) for v, n in zip(vals, counts))
            major = int(vals[np.argmax(counts)])
            row.append(names.get(major, "?"))
            if len(pairs) > 1:
                mixed.append(((i, j), pairs))
        rows.append("".join(row))
    return rows, mixed


def probe(env):
    solver.solve(env)
    if env.levels_completed != 4:
        print("arrival", env.levels_completed)
        return
    print("actions", tuple(env.actions))
    print("colors", color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    print("coarse", *coarse(env.frame()), sep="\n")
    tile_rows, mixed = tiles(env.frame())
    print("tiles", *tile_rows, sep="\n")
    print("mixed_tiles", mixed)
    for action, delta in action_deltas(env).items():
        child = env.clone()
        child.step(action)
        print(
            ACTION_NAME[action],
            "delta", (delta["count"], delta["bbox"]),
            "level", child.levels_completed,
            "trans", transitions(env.frame(), child.frame()),
            "blobs", compact_blobs(child.frame()),
        )
    reward_path, reach = fast_reach(env)
    print("reach", "reward", reward_path, "count", len(reach))
    print("positions", sorted((p, len(path), path) for p, path in reach.items()))
    for pos, path in sorted(reach.items(), key=lambda item: (len(item[1]), item[0])):
        node = clone_after(env, path)
        used = clone_after(node, 5)
        delta = action_deltas(node, (5,))[5]
        rp2, reach2 = fast_reach(used)
        before_sig = block_signatures(env.frame(), cell=6)
        after_sig = block_signatures(used.frame(), cell=6)
        changed = tuple(k for k in sorted(before_sig) if before_sig[k] != after_sig[k])
        print("use_at", pos, path, "delta", (delta["count"], delta["bbox"]),
              "avatar", _avatar_pos(used.frame()), "level", used.levels_completed,
              "next", (len(reach2), rp2), "persist", changed)


levels, path, err = A.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
