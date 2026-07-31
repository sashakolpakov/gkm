import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


N1 = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
NN = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def click(env, mode):
    color = 6 if mode == "h" else 15
    blobs = [
        blob
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=16
        )
        if blob.centroid[0] < 53
    ]
    blob = min(
        blobs,
        key=lambda item: item.centroid[1] if mode == "h"
        else item.centroid[0],
    )
    return (6, round(blob.centroid[1]), round(blob.centroid[0]))


def apply(env, path):
    for action in path:
        if action in ("h", "v"):
            env.step(*click(env, action))
        else:
            env.step(action)


def positions(env, color):
    return tuple(sorted(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=12
        )
        if blob.centroid[0] < 53
    ))


def horizontal_row(env):
    blob = click(env, "h")
    return blob[2]


def probe(env):
    reach_level_6(env)
    clean9 = (
        ["v"] + N1
        + ["v"] + NN + ["h", 4, "v", 1, 1, 1]
        + ["v"] + NN + ["h", 4, "v", 1, 1, 1]
        + ["v", 4, "h", 2]
    )
    base = env.clone()
    apply(base, clean9)
    print("BASE", horizontal_row(base), positions(base, 8), positions(base, 9))
    for extend in range(7):
        for retract in range(7):
            node = base.clone()
            apply(node, [4] * extend + [3] * retract)
            before8 = positions(node, 8)
            before9 = positions(node, 9)
            before_row = horizontal_row(node)
            apply(node, [1])
            after8 = positions(node, 8)
            after9 = positions(node, 9)
            after_row = horizontal_row(node)
            if after8 != before8:
                print(
                    "HOOK", extend, retract,
                    "row", (before_row, after_row),
                    "8", before8, "->", after8,
                    "9", before9, "->", after9,
                )


A.run_program("sk48", probe)
