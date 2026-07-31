"""Compact raw-observation probe for g50t level 6."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p


spec = importlib.util.spec_from_file_location("workspace_solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def compact_delta(delta):
    transitions = {}
    for _, _, before, after in delta["samples"]:
        transitions[(before, after)] = transitions.get((before, after), 0) + 1
    return {
        "count": delta["count"],
        "bbox": delta["bbox"],
        "sample_transitions": transitions,
    }


def tracked(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in p.connected_components(frame, colors=(8, 9, 11, 14), min_area=4)
    ]


def tile_map(frame):
    f = p.arr(frame)
    names = {0: ".", 5: "#", 8: "8", 9: "9", 11: "B", 14: "E"}
    rows = []
    for r in range(8, 57, 6):
        row = []
        for c in range(2, 57, 6):
            vals, cnts = __import__("numpy").unique(f[r:r + 5, c:c + 5],
                                                    return_counts=True)
            color = int(vals[int(cnts.argmax())])
            row.append(names.get(color, str(color)))
        rows.append("".join(row))
    return rows


def observe_level_6(env):
    print("L6", {
        "level": int(env.levels_completed) + 1,
        "actions": list(env.actions),
        "counts": p.color_counts(env.frame()),
    })
    objects = [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(env.frame(), min_area=4)
    ]
    print("OBJECTS", objects)
    print("ACTIONS", {
        a: compact_delta(d)
        for a, d in p.action_deltas(env, tuple(env.actions)).items()
    })
    print("TILES", tile_map(env.frame()))
    paths = [
        [1], [2], [3], [4], [5],
        [2, 2], [3, 3], [2, 3], [3, 2],
        [2, 2, 2], [3, 3, 3],
        [2, 2, 2, 3], [2, 2, 2, 3, 3],
        [3, 3, 2, 3, 3, 2, 2, 4, 4],
    ]
    for path in paths:
        node = p.replay(env, path)
        print("PATH", path, {
            "level": int(node.levels_completed),
            "tracked": tracked(node.frame()),
        })
    for prefix in ([], [3], [3, 3], [3, 3, 2, 3, 3],
                   [3, 3, 2, 3, 3, 2]):
        node = p.replay(env, prefix)
        after = p.replay(node, [5])
        print("USE_AT", prefix, {
            "delta": compact_delta(p.frame_delta(node.frame(), after.frame())),
            "tracked_before": tracked(node.frame()),
            "tracked_after": tracked(after.frame()),
        })
    for n in range(1, 9):
        path = [a for _ in range(n) for a in (3, 4)]
        node = p.replay(env, path)
        print("TICK", len(path), tracked(node.frame()))


solver.players.play_level_6 = observe_level_6


def resumed_solver(env):
    checkpoint = None
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as handle:
            checkpoint = json.load(handle)
    if (checkpoint and checkpoint.get("game") == "g50t"
            and checkpoint.get("validated") and checkpoint.get("final_path")):
        for action in checkpoint["final_path"]:
            env.step(action)
    solver.solve(env)


levels, path, err = arena.run_program("g50t", resumed_solver)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "err": str(err)})
