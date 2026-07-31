import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _avatar_pos
from perception import action_deltas, color_counts, connected_components


def summary(env):
    blobs = connected_components(
        env.frame(), colors=(8, 9, 11, 14, 15), min_area=4
    )
    return (
        int(env.levels_completed),
        bool(env.terminal()),
        _avatar_pos(env.frame()),
        tuple((b.color, b.bbox, b.area) for b in blobs),
    )


def tile_map(frame):
    chars = {0: ".", 1: "1", 5: "#", 8: "G", 9: "A",
             11: "S", 14: "B", 15: "X"}
    return "/".join(
        "".join(chars.get(int(frame[r + 2][c + 2]), "?")
                for c in range(2, 62, 6))
        for r in range(2, 62, 6)
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint:
        env.step(action)
    print("base", env.actions, color_counts(env.frame()), summary(env))
    print("map", tile_map(env.frame()))
    deltas = action_deltas(env, env.actions)
    for action in env.actions:
        child = env.clone()
        child.step(action)
        delta = deltas[action]
        print(
            "act",
            action,
            (delta["count"], delta["bbox"]),
            summary(child),
        )


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
