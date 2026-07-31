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


def click(env, mode):
    color = 6 if mode == "h" else 15
    heads = [
        blob
        for blob in connected_components(env.frame(), colors=(color,), min_area=16)
        if blob.centroid[0] < 53
    ]
    head = min(
        heads,
        key=lambda blob: blob.centroid[1] if mode == "h" else blob.centroid[0],
    )
    return (6, round(head.centroid[1]), round(head.centroid[0]))


def apply(env, path):
    for action in path:
        if action in ("h", "v"):
            env.step(*click(env, action))
        else:
            env.step(action)
        if env.levels_completed > 5:
            return


def positions(env, color):
    return tuple(
        sorted(
            (round(blob.centroid[0]), round(blob.centroid[1]))
            for blob in connected_components(env.frame(), colors=(color,), min_area=12)
            if blob.centroid[0] < 53
        )
    )


def distance(points, targets):
    if len(points) != len(targets):
        return 100
    return min(
        sum(
            abs(row - target_row) + abs(col - target_col)
            for (row, col), (target_row, target_col) in zip(points, ordering)
        )
        for ordering in itertools.permutations(targets)
    ) // 6


def score(env):
    return sum(distance(positions(env, color), TARGETS[color]) for color in (8, 9))


def schedules():
    for e_slots in itertools.combinations(range(6), 3):
        e_slots = set(e_slots)
        ei = ni = 0
        schedule = []
        for index in range(6):
            if index in e_slots:
                ei += 1
                schedule.append(f"E{ei}")
            else:
                ni += 1
                schedule.append(f"N{ni}")
        yield tuple(schedule)


def correction(mode, amount):
    if mode == "h":
        action = 2 if amount > 0 else 1
    else:
        action = 4 if amount > 0 else 3
    return [mode] + [action] * abs(amount)


def probe(env):
    reach_level_6(env)
    finish = env.clone()
    for name in ("E1", "E2", "E3", "N1", "N2", "N3"):
        apply(finish, CHUNKS[name])
    apply(finish, ["h", 4, 4])
    print("FOCUS", positions(finish, 8), positions(finish, 9))
    cases = {
        "top_pair": ["h", 1, 1],
        "top_pair_v2": ["h", 1, 1, "v", 2],
        "top_pair_v2_h3": ["h", 1, 1, "v", 2, "h", 3],
        "top_pair_v2_h333": ["h", 1, 1, "v", 2, "h", 3, 3, 3],
        "top_pair_v2_h333_v4": [
            "h", 1, 1, "v", 2, "h", 3, 3, 3, "v", 4
        ],
        "top_pair_transfer_v2": [
            "h", 1, 1, "v", 2, "h", 3, 3, 3, "v", 2, 2
        ],
        "top_pair_transfer_v2_right": [
            "h", 1, 1, "v", 2, "h", 3, 3, 3, "v", 2, 2, 4
        ],
        "top_pair_transfer_v1": [
            "h", 1, 1, "v", 2, "h", 3, 3, 3, "v", 1, 1
        ],
    }
    transfer_base = [
        "h", 1, 1, "v", 2, "h", 3, 3, 3, "v", 2, 2
    ]
    for count in range(1, 6):
        cases[f"clear{count}_right"] = transfer_base + ["v"] + [1] * count + [4]
    clear_one = transfer_base + ["v", 1, 4]
    for action in (3, 4):
        for count in range(1, 7):
            cases[f"clear1_h{action}x{count}"] = (
                clear_one + ["h"] + [action] * count
            )
    for count in range(1, 7):
        cases[f"clear1_hook_pull{count}"] = (
            clear_one + ["h"] + [4] * 4 + [3] * count
        )
    for name, path in cases.items():
        node = finish.clone()
        apply(node, path)
        print("CASE", name, positions(node, 8), positions(node, 9))
    for mode in ("h", "v"):
        for action in (1, 2, 3, 4):
            node = finish.clone()
            snapshots = []
            for count in range(1, 7):
                apply(node, [mode, action])
                state = (positions(node, 8), positions(node, 9))
                if not snapshots or state != snapshots[-1][1]:
                    snapshots.append((count, state))
            print("RUN", mode, action, snapshots)
    return
    results = []
    for schedule in schedules():
        base = env.clone()
        for name in schedule:
            apply(base, CHUNKS[name])
        for h_shift in range(-2, 3):
            for v_shift in range(-2, 3):
                for order in ("hv", "vh"):
                    node = base.clone()
                    shifts = {"h": h_shift, "v": v_shift}
                    for mode in order:
                        if shifts[mode]:
                            apply(node, correction(mode, shifts[mode]))
                    item = (
                        score(node),
                        -node.levels_completed,
                        schedule,
                        h_shift,
                        v_shift,
                        order,
                        positions(node, 8),
                        positions(node, 9),
                    )
                    results.append(item)
                    if node.levels_completed > 5:
                        print("WIN", item)
                        return
    for item in sorted(results)[:12]:
        print("BEST", item)


A.run_program("sk48", probe)
