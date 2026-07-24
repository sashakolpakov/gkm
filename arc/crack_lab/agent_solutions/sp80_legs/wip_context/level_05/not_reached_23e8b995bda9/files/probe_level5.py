"""Compact observational probes for sp80 level 5."""
import importlib.util
import random
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import (
    action_deltas,
    arr,
    color_counts,
    connected_components,
    frame_delta,
    object_candidates,
)


def load_players():
    spec = importlib.util.spec_from_file_location("players", "players.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame, min_area=4)
    ]


def pieces(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame, min_area=4)
        if o["color"] in (4, 6, 8, 9, 11, 15)
    ]


def component_masks(frame):
    f = arr(frame)
    for blob in connected_components(f, colors=(8, 9, 11, 15), min_area=4):
        r0, c0, r1, c1 = blob.bbox
        mask = f[r0:r1 + 1, c0:c1 + 1] == blob.color
        rows = tuple("".join("#" if v else "." for v in row) for row in mask)
        print("MASK", blob.color, blob.bbox, rows)


def macro_map(frame):
    f = arr(frame)
    symbols = {1: "#", 4: "g", 6: "G", 8: "b", 9: "B",
               11: "S", 12: ".", 15: "L"}
    for r in range(5, 62, 3):
        row = []
        for c in range(5, 62, 3):
            vals, counts = __import__("numpy").unique(
                f[r:min(r + 3, 64), c:min(c + 3, 64)], return_counts=True
            )
            row.append(symbols.get(int(vals[counts.argmax()]), "?"))
        print("MAP", f"{r:02}", "".join(row))
    for r in range(5, 62, 3):
        row = []
        for c in range(5, 62, 3):
            count = int((f[r:min(r + 3, 64), c:min(c + 3, 64)] == 1).sum())
            row.append("." if count == 0 else str(min(count, 9)))
        print("WALL", f"{r:02}", "".join(row))
    print("COLOR1_ROWS", tuple((i, int((f[i] == 1).sum()))
                               for i in range(64) if (f[i] == 1).any()))
    print("COLOR1_COLS", tuple((i, int((f[:, i] == 1).sum()))
                               for i in range(64) if (f[:, i] == 1).any()))


def move_axis(env, delta, negative, positive):
    action = negative if delta < 0 else positive
    for _ in range(abs(delta) // 3):
        env.step(action)


def search_columns(env, trials=1200):
    rng = random.Random(805)
    choices = (
        ("A12", 34, 21, 29, tuple(range(5, 51, 3))),
        ("B9", 24, 33, 20, tuple(range(14, 54, 3))),
        ("C15", None, None, 41, tuple(range(14, 48, 3))),
        ("D_L", 34, 43, 32, tuple(range(14, 57, 3))),
    )
    for trial in range(trials):
        node = env.clone()
        targets = {}
        for name, x, y, start, values in choices:
            target = rng.choice(values)
            targets[name] = target
            if name == "C15":
                f = arr(node.frame())
                ys, xs = __import__("numpy").where(f[32:35, 41:56] == 8)
                if not len(xs):
                    break
                x, y = int(xs[0] + 41), int(ys[0] + 32)
            node.step(6, x, y)
            move_axis(node, target - start, 3, 4)
        else:
            node.step(5)
            if node.levels_completed > env.levels_completed:
                print("COLUMN_WIN", trial, targets)
                return targets
    print("COLUMN_WIN", None, trials)
    return None


def probe(env):
    players = load_players()
    for level in range(1, 4):
        getattr(players, f"play_level_{level}")(env)
    print("REFERENCE_L4", [
        o for o in compact_objects(env.frame()) if o[0] in (4, 6, 8, 9, 11, 15)
    ])
    players.play_level_4(env)

    print("LEVEL", env.levels_completed)
    print("ACTIONS", env.actions)
    print("COLORS", color_counts(env.frame()))
    print("OBJECTS", compact_objects(env.frame()))
    component_masks(env.frame())
    macro_map(env.frame())
    deltas = action_deltas(env, actions=(1, 2, 3, 4, 5))
    print("KEY_DELTAS", {a: (d["count"], d["bbox"]) for a, d in deltas.items()})
    for action in (1, 2, 3, 4, 5):
        clone = env.clone()
        clone.step(action)
        print("AFTER_KEY", action, pieces(clone.frame()))

    base = env.frame()
    for color, bbox, area in compact_objects(base):
        r0, c0, r1, c1 = bbox
        x, y = (c0 + c1) // 2, (r0 + r1) // 2
        clone = env.clone()
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        print("CLICK", (color, bbox, area), (x, y),
              (delta["count"], delta["bbox"]),
              pieces(clone.frame()))

    clone = env.clone()
    clone.step(6, 34, 43)
    clone.step(6, 34, 21)
    print("RESELECT_AFTER_15", pieces(clone.frame()))

    for name, x, y in (
        ("A12", 34, 21),
        ("B9", 24, 33),
        ("C15", 48, 33),
        ("D_L", 34, 43),
    ):
        limits = {}
        for action in (1, 2, 3, 4):
            clone = env.clone()
            clone.step(6, x, y)
            for _ in range(30):
                clone.step(action)
            selected = [p for p in pieces(clone.frame()) if p[0] == 9]
            limits[action] = selected
        print("LIMITS", name, limits)
    search_columns(env)


if __name__ == "__main__":
    levels, path, err = arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
