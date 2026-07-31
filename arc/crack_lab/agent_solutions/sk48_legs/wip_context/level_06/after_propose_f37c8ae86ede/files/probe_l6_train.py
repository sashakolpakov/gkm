import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]


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
        if env.levels_completed > 5:
            return


def positions(env, color):
    return tuple(sorted(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=12
        )
        if blob.centroid[0] < 53
    ))


def state(env):
    if env.levels_completed > 5:
        return ("WIN", env.levels_completed)
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
    home_h = [2, 4]
    pick1 = home_h + ["v", 4]
    carry1 = pick1 + [3] * 4
    cases = {
        "full8": [],
        "home_h": home_h,
        "pick1": pick1,
        "pick1_down_left": pick1 + [2, 3],
        "pick1_cycle_left": pick1 + [2, 1, 3],
        "pick1_down2_left": pick1 + [2, 2, 3],
        "pick1_down2_up_left": pick1 + [2, 2, 1, 3],
        "pick1_up_left": pick1 + [1, 3],
        "carry1": carry1,
        "split_select": carry1 + ["h", "v", 4],
        "split_h3": carry1 + ["h", 3, "v", 4],
        "split_h33": carry1 + ["h", 3, 3, "v", 4],
        "split_h4": carry1 + ["h", 4, "v", 4],
        "split_h43": carry1 + ["h", 4, 3, "v", 4],
    }
    for retract in range(7):
        cases[f"wall_hook_r{retract}"] = (
            pick1 + [2] * 3 + [1] * retract + [3]
        )
    for name, suffix in cases.items():
        node = env.clone()
        try:
            apply(node, full8 + suffix)
            print(name, len(full8 + suffix), state(node))
        except Exception as exc:
            print(name, "ERROR", type(exc).__name__, str(exc))
    lowered = (
        pick1 + [2] * 3 + [1] * 3 + [3]
        + ["h"] + [2] * 3
    )
    desired8 = ((10, 32), (16, 32), (22, 32))
    desired9 = ((28, 14), (28, 44), (28, 50))
    trials = []
    for extend, retract in ((5, 6),):
        suffix = (
            lowered + [4] * extend + [3] * retract + [1] * 3
        )
        node = env.clone()
        apply(node, full8 + suffix)
        points8 = positions(node, 8)
        points9 = positions(node, 9)
        score = (
            sum(
                abs(r - tr) + abs(c - tc)
                for (r, c), (tr, tc) in zip(points8, desired8)
            )
            + sum(
                abs(r - tr) + abs(c - tc)
                for (r, c), (tr, tc) in zip(points9, desired9)
            )
        ) // 6
        trials.append((score, extend, retract, points8, points9))
    for item in sorted(trials)[:10]:
        print("LOWERED", item)
    first_done = lowered + [4] * 5 + [3] * 6 + [1] * 3
    second_lowered = (
        first_done + [3, "v"] + [4] * 2
        + [2] * 3 + [1] * 3 + [3] * 2
        + ["h"] + [2] * 3
    )
    desired9_second = ((28, 14), (28, 20), (28, 50))
    second_trials = []
    for extend in range(4, 7):
        for retract in range(4, 7):
            suffix = (
                second_lowered + [4] * extend
                + [3] * retract + [1] * 3
            )
            node = env.clone()
            apply(node, full8 + suffix)
            points8 = positions(node, 8)
            points9 = positions(node, 9)
            score = (
                sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc) in zip(points8, desired8)
                )
                + sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc)
                    in zip(points9, desired9_second)
                )
            ) // 6
            second_trials.append(
                (score, extend, retract, points8, points9)
            )
    for item in sorted(second_trials):
        print("SECOND", item)
    second_loaded = (
        first_done + ["v"] + [4] * 2
        + [2] * 3 + [1] * 3 + [3] * 2
        + ["h"] + [2] * 3
    )
    loaded_trials = []
    for extend in range(3, 8):
        for retract in range(3, 8):
            suffix = (
                second_loaded + [4] * extend
                + [3] * retract + [1] * 3
            )
            node = env.clone()
            apply(node, full8 + suffix)
            points8 = positions(node, 8)
            points9 = positions(node, 9)
            score = (
                sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc) in zip(points8, desired8)
                )
                + sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc)
                    in zip(points9, desired9_second)
                )
            ) // 6
            loaded_trials.append(
                (score, extend, retract, points8, points9)
            )
    for item in sorted(loaded_trials)[:10]:
        print("LOADED2", item)
    second_done = (
        second_loaded + [4] * 6 + [3] * 5 + [1] * 3
    )
    third_loaded = (
        second_done + ["v"] + [4] * 3
        + [2] * 3 + [1] * 3 + [3] * 3
        + ["h"] + [2] * 3
    )
    third_trials = []
    for extend in range(5, 9):
        for retract in range(3, 7):
            suffix = (
                third_loaded + [4] * extend
                + [3] * retract + [1] * 3
            )
            node = env.clone()
            apply(node, full8 + suffix)
            if node.levels_completed > 5:
                print(
                    "THIRD_WIN", extend, retract,
                    "length", len(full8 + suffix),
                )
                continue
            points8 = positions(node, 8)
            points9 = positions(node, 9)
            score = (
                sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc) in zip(points8, desired8)
                )
                + sum(
                    abs(r - tr) + abs(c - tc)
                    for (r, c), (tr, tc)
                    in zip(points9, ((28, 14), (28, 20), (28, 26)))
                )
            ) // 6
            third_trials.append(
                (score, extend, retract, points8, points9)
            )
    for item in sorted(third_trials)[:10]:
        print("THIRD", item)


A.run_program("sk48", probe)
