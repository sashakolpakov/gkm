import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


FIRST = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
TRANSFER = ["v", 2, 1]
NEXT = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
FIRST_NINE = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
NEXT_NINE = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]


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


def positions(env, color=8):
    return tuple(
        sorted(
            (round(blob.centroid[0]), round(blob.centroid[1]))
            for blob in connected_components(env.frame(), colors=(color,), min_area=12)
            if blob.centroid[0] < 53
        )
    )


def ownership(env, color=8):
    base = positions(env, color)
    results = []
    for mode, action in (("h", 1), ("h", 2), ("v", 3), ("v", 4)):
        node = env.clone()
        apply(node, [mode, action])
        after = positions(node, color)
        if after != base:
            results.append((mode + str(action), after))
    return results


def report(root, name, path):
    node = root.clone()
    apply(node, path)
    print(
        name,
        "L",
        node.levels_completed,
        "P8",
        positions(node, 8),
        "O8",
        ownership(node, 8),
        "P9",
        positions(node, 9),
        "O9",
        ownership(node, 9),
    )


def probe(env):
    reach_level_6(env)
    clean_nine1 = ["v"] + FIRST_NINE
    clean_nine2 = clean_nine1 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    clean_nine3 = clean_nine2 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    second_kept = (
        FIRST + TRANSFER + ["h"] + NEXT + ["v", 2, "h", 3, 3, 3]
    )
    third_staged = second_kept + ["h"] + NEXT
    third_kept = third_staged + ["v", 2, "h", 3, 3, 3]
    parked_eights = third_kept + ["v", 1, 1, 1]
    first_nine_staged = parked_eights + FIRST_NINE
    second_nine_staged = first_nine_staged + NEXT_NINE
    second_nine_kept = second_nine_staged + ["h", 4, "v", 1, 1, 1]
    third_nine_staged = second_nine_kept + NEXT_NINE
    third_nine_kept = third_nine_staged + ["h", 4, "v", 1, 1, 1]
    e_then_n1 = third_kept + ["v"] + FIRST_NINE
    e_then_n2 = e_then_n1 + ["v"] + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    e_then_n3 = e_then_n2 + ["v"] + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    states = {
        "clean_nine1": clean_nine1,
        "clean_nine2": clean_nine2,
        "clean_nine3": clean_nine3,
        "first_staged": FIRST,
        "first_transfer": FIRST + TRANSFER,
        "second_staged": FIRST + TRANSFER + ["h"] + NEXT,
        "second_top_extend": FIRST + TRANSFER + ["h"] + NEXT + ["v", 2],
        "second_left_retract3": (
            FIRST + TRANSFER + ["h"] + NEXT + ["v", 2, "h", 3, 3, 3]
        ),
        "second_final": (
            FIRST + TRANSFER + ["h"] + NEXT
            + ["v", 2, "h", 3, 3, 3, "v", 1]
        ),
        "third_staged_keep": third_staged,
        "third_transfer_keep": third_kept,
        "eights_drop1": third_kept + ["v", 1],
        "eights_drop2": third_kept + ["v", 1, 1],
        "eights_drop3": parked_eights,
        "eights_drop4": parked_eights + [1],
        "eights_drop5": parked_eights + [1, 1],
        "eights_drop6": parked_eights + [1, 1, 1],
        "eights_drop3_shift": parked_eights + [4],
        "eights_drop4_shift": parked_eights + [1, 4],
        "eights_drop5_shift": parked_eights + [1, 1, 4],
        "eights_drop6_shift": parked_eights + [1, 1, 1, 4],
        "drop4_first_nine": parked_eights + [1] + FIRST_NINE,
        "drop5_first_nine": parked_eights + [1, 1] + FIRST_NINE,
        "drop6_first_nine": parked_eights + [1, 1, 1] + FIRST_NINE,
        "first_nine_staged": first_nine_staged,
        "second_nine_staged": second_nine_staged,
        "second_nine_kept": second_nine_kept,
        "third_nine_staged": third_nine_staged,
        "third_nine_kept": third_nine_kept,
        "return_v": third_nine_kept + ["v", 4],
        "return_v_acquire1": third_nine_kept + ["v", 4, 2],
        "return_v_acquire2": third_nine_kept + ["v", 4, 2, 2],
        "return_v_acquire3": third_nine_kept + ["v", 4, 2, 2, 2],
        "e_then_n1": e_then_n1,
        "e_then_n2": e_then_n2,
        "e_then_n3": e_then_n3,
    }
    for name, path in states.items():
        report(env, name, path)


A.run_program("sk48", probe)
