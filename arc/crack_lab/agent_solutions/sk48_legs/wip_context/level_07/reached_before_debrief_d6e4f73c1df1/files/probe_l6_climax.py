import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
N1 = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3


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


def prefix():
    full8 = (
        E1 + ["v", 2, 1, "h"] + EN + ["v", 2, "h", 3, 3, 3]
        + ["h"] + EN + ["v", 2, "h", 3, 3, 3]
    )
    park = (
        full8 + ["v", 1, 1, 1, "h"]
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5 + [1]
        + [4] * 5 + [3] * 5
        + ["h", 2, 2, 2, 4, "v", 2] + N1
    )
    n2_stage = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4
    n2_fix = ["h", 3, "v", 2, "h", 4, "v", 1, 1, 1]
    return park + n2_stage + n2_fix


def probe(env):
    reach_level_6(env)
    root = prefix()
    restore_setup = [4, 4, 2, "h", 3, 1]
    restore_one = restore_setup + [4] * 6 + [3] * 3
    keep_one = restore_one + ["v", 2, "h", 3, 3, 3]
    restore_two = keep_one + [1] + [4] * 6 + [3] * 4
    keep_two = restore_two + ["v", 2, "h", 3, 3, 3]
    restore_three = keep_two + [1] + [4] * 6 + [3] * 4
    keep_three = restore_three + ["v", 2, "h", 3, 3, 3]
    join_two = restore_two + ["v", 4, 3]
    pull_three = join_two + ["h", 1] + [4] * 6 + [3] * 4
    join_three = pull_three + ["v", 4, 3]
    return_horizontal = join_three + ["h", 2, 2, 2]
    td_setup = [4, 4, 2, "h", 3, 1, 1, 1]
    td_pull1 = td_setup + [4] * 6 + [3] * 3
    td_keep1 = td_pull1 + ["v", 2, 1]
    td_pull2 = td_keep1 + ["h", 2] + [4] * 6 + [3] * 3
    td_keep2 = td_pull2 + ["v", 2, "h", 3, 3, 3]
    td_pull3 = td_keep2 + [2] + [4] * 6 + [3] * 3
    td_keep3 = td_pull3 + ["v", 2, "h", 3, 3, 3]
    td2_setup = [4, 4, 2, "h", 3, 3, 1, 1, 1]
    td2_pull1 = td2_setup + [4] * 6 + [3] * 3
    td2_keep1 = td2_pull1 + ["v", 2, 1]
    td2_pull2 = td2_keep1 + ["h", 2] + [4] * 6 + [3] * 3
    td2_keep2 = td2_pull2 + ["v", 2, "h", 3, 3, 3]
    td2_pull3 = td2_keep2 + [2] + [4] * 6 + [3] * 3
    td2_keep3 = td2_pull3 + ["v", 2, "h", 3, 3, 3]
    td3_setup = [4, 4, "h", 3, 3, 1, 1, 1]
    td3_pull1 = td3_setup + [4] * 6 + [3] * 3
    td3_keep1 = td3_pull1 + ["v", 2, 1]
    td3_pull2 = td3_keep1 + ["h", 2] + [4] * 6 + [3] * 3
    td3_keep2 = td3_pull2 + ["v", 2, "h", 3, 3, 3]
    td3_pull3 = td3_keep2 + [2] + [4] * 6 + [3] * 3
    td3_keep3 = td3_pull3 + ["v", 2, "h", 3, 3, 3]
    td4_setup = ["h", 3, 3, 1, 1, 1]
    td4_pull1 = td4_setup + [4] * 6 + [3] * 3
    td4_top1 = td4_pull1 + ["v", 4]
    td4_top2 = td4_pull1 + ["v", 4, 4]
    cases = {
        "ready": [],
        "td4_setup": td4_setup,
        "td4_pull1": td4_pull1,
        "td4_top1": td4_top1,
        "td4_top2": td4_top2,
    }
    for name, suffix in cases.items():
        node = env.clone()
        try:
            apply(node, root + suffix)
            print(name, len(root + suffix), state(node))
        except Exception as exc:
            print(name, "ERROR", type(exc).__name__, str(exc))


A.run_program("sk48", probe)
