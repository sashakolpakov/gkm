import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
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


def state(env):
    h = click(env, "h")
    v = click(env, "v")
    return (
        (h[2], h[1]), (v[2], v[1]),
        positions(env, 8), positions(env, 9),
        env.levels_completed,
    )


def probe(env):
    reach_level_6(env)
    full8 = (
        E1 + ["v", 2, 1, "h"] + EN + ["v", 2, "h", 3, 3, 3]
        + ["h"] + EN + ["v", 2, "h", 3, 3, 3]
    )
    detached8 = full8 + ["v", 1, 1, 1, "h"]
    park_right = (
        detached8
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5
    )
    cases = {
        "clean_n1": ["v"] + N1,
        "clean_n2_stage": ["v"] + N1 + NN,
        "clean_n2_accept": ["v"] + N1 + NN + ["h", 4],
        "clean_n12": (
            ["v"] + N1 + NN + ["h", 4, "v", 1, 1, 1]
        ),
        "full8": full8,
        "detached8": detached8,
        "push_once": detached8 + [4],
        "push_twice": detached8 + [4, 4],
        "push_cycle1": detached8 + [4, 3],
        "push_cycle2": detached8 + [4, 3] * 2,
        "push_cycle3": detached8 + [4, 3] * 3,
        "push_cycle4": detached8 + [4, 3] * 4,
        "push_step_up": detached8 + [4, 1],
        "push_step_up_retract": detached8 + [4, 1, 3],
        "push_step_reset": detached8 + [4, 1, 3, 2],
        "push_step_twice": detached8 + [4, 1, 3, 2, 4],
        "park_right": park_right,
        "park_right_home": park_right + ["h", 2, 2, 2, 4],
        "park_right_n1": (
            park_right + ["h", 2, 2, 2, 4, "v", 2] + N1
        ),
        "park_right_n2_stage": (
            park_right + ["h", 2, 2, 2, 4, "v", 2] + N1
            + NN
        ),
        "park_right_n2_accept": (
            park_right + ["h", 2, 2, 2, 4, "v", 2] + N1
            + NN + ["h", 4]
        ),
        "park_right_n12": (
            park_right + ["h", 2, 2, 2, 4, "v", 2] + N1
            + NN
            + ["h", 4, "v", 1, 1, 1]
        ),
    }
    for name, path in cases.items():
        node = env.clone()
        apply(node, path)
        print(name, len(path), state(node))
    parked_n1 = (
        park_right + ["h", 2, 2, 2, 4, "v", 2] + N1
    )
    trials = []
    for lift in range(2, 6):
        for final_drop in range(3):
            path = (
                parked_n1 + [4] * 5 + [2] * 6 + [1] * lift
                + [3] * 4 + [2] * final_drop
            )
            node = env.clone()
            apply(node, path)
            target_hits = len(
                set(positions(node, 9)) & {(28, 14), (28, 20)}
            )
            trials.append(
                (-target_hits, lift, final_drop, len(path), state(node))
            )
    for item in sorted(trials)[:8]:
        print("N2TRY", item)
    for retract in range(8):
        node = env.clone()
        apply(node, park_right + [4] * 6 + [3] * retract)
        print("PULLTRY", retract, state(node))
    n2_fixed = (
        parked_n1 + [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4
        + ["h", 3, "v", 2]
    )
    direct = []
    target9 = ((28, 14), (28, 20), (28, 26))
    for extend in range(9):
        for retract in range(9):
            path = n2_fixed + ["h"] + [4] * extend + [3] * retract
            node = env.clone()
            apply(node, path)
            distance = sum(
                abs(r - tr) + abs(c - tc)
                for (r, c), (tr, tc)
                in zip(positions(node, 9), target9)
            ) // 6
            direct.append((distance, extend, retract, len(path), state(node)))
    for item in sorted(direct)[:10]:
        print("DIRECT", item)
    for retract in range(5):
        node = env.clone()
        suffix = ["h", 4, "v"] + [1] * retract
        apply(node, n2_fixed + suffix)
        print("N2RELEASE", retract, len(n2_fixed + suffix), state(node))


A.run_program("sk48", probe)
