import itertools
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
N1 = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
NN = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]
CHUNKS = {
    "E1": ["h"] + E1,
    "E2": ["h"] + EN + ["v", 2, "h", 3, 3, 3],
    "E3": ["h"] + EN + ["v", 2, "h", 3, 3, 3],
    "N1": ["v"] + N1,
    "N2": ["v"] + NN + ["h", 4, "v", 1, 1, 1],
    "N3": ["v"] + NN + ["h", 4, "v", 1, 1, 1],
}
TARGETS = {
    8: ((10, 32), (16, 32), (22, 32)),
    9: ((28, 14), (28, 20), (28, 26)),
}


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def head(env, mode):
    color = 6 if mode == "h" else 15
    blobs = [
        blob
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=16
        )
        if blob.centroid[0] < 53
    ]
    return min(
        blobs,
        key=lambda blob: blob.centroid[1] if mode == "h"
        else blob.centroid[0],
    )


def click(env, mode):
    blob = head(env, mode)
    return (6, round(blob.centroid[1]), round(blob.centroid[0]))


def apply(env, path):
    count = 0
    for action in path:
        if action in ("h", "v"):
            env.step(*click(env, action))
        else:
            env.step(action)
        count += 1
        if env.levels_completed > 5:
            break
    return count


def positions(env, color):
    return tuple(sorted(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=12
        )
        if blob.centroid[0] < 53
    ))


def distance(env):
    total = 0
    for color, targets in TARGETS.items():
        points = positions(env, color)
        if len(points) != len(targets):
            return 999
        total += min(
            sum(
                abs(row - target_row) + abs(col - target_col)
                for (row, col), (target_row, target_col)
                in zip(points, ordering)
            )
            for ordering in itertools.permutations(targets)
        ) // 6
    return total


def schedules():
    for e_slots in itertools.combinations(range(6), 3):
        e_slots = set(e_slots)
        ei = ni = 0
        names = []
        for index in range(6):
            if index in e_slots:
                ei += 1
                names.append(f"E{ei}")
            else:
                ni += 1
                names.append(f"N{ni}")
        yield tuple(names)


def home_path(env, order):
    paths = {}
    horizontal = head(env, "h")
    h_steps = (28 - round(horizontal.centroid[0])) // 6
    paths["h"] = ["h"] + ([2] * h_steps if h_steps > 0
                           else [1] * -h_steps)
    vertical = head(env, "v")
    v_steps = (32 - round(vertical.centroid[1])) // 6
    paths["v"] = ["v"] + ([4] * v_steps if v_steps > 0
                           else [3] * -v_steps)
    return paths[order[0]] + paths[order[1]]


def probe(env):
    reach_level_6(env)
    results = []
    for names in schedules():
        base = env.clone()
        path = []
        for name in names:
            path.extend(CHUNKS[name])
        apply(base, path)
        for order in ("hv", "vh"):
            node = base.clone()
            suffix = home_path(node, order)
            apply(node, suffix)
            item = (
                distance(node),
                names,
                order,
                positions(node, 8),
                positions(node, 9),
                len(path) + len(suffix),
                node.levels_completed,
            )
            results.append(item)
            if node.levels_completed > 5:
                print("WIN", item)
                return
    for item in sorted(results)[:20]:
        print("BEST", item)


A.run_program("sk48", probe)
