import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


FIRST_EIGHT = (
    [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
)
NEXT_EIGHT = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
FIRST_NINE = (
    [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
)
NEXT_NINE = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def apply(env, path):
    for action in path:
        if action in ("h", "v"):
            env.step(*selection(env, action))
        elif isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)


def selection(env, mode):
    color = 6 if mode == "h" else 15
    heads = [
        blob
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=16
        )
        if blob.centroid[0] < 53
    ]
    head = min(
        heads,
        key=lambda blob: blob.centroid[1] if mode == "h"
        else blob.centroid[0],
    )
    return (6, round(head.centroid[1]), round(head.centroid[0]))


def live_positions(env, color):
    return tuple(
        sorted(
            (round(blob.centroid[0]), round(blob.centroid[1]))
            for blob in connected_components(
                env.frame(), colors=(color,), min_area=12
            )
            if blob.centroid[0] < 53
        )
    )


def head_positions(env):
    result = []
    for color in (6, 15):
        blobs = [
            blob
            for blob in connected_components(
                env.frame(), colors=(color,), min_area=16
            )
            if blob.centroid[0] < 53
        ]
        blob = min(
            blobs,
            key=lambda item: item.centroid[1] if color == 6
            else item.centroid[0],
        )
        result.append((round(blob.centroid[0]), round(blob.centroid[1])))
    return tuple(result)


def follows(env, color, selection, movement):
    before = live_positions(env, color)
    node = env.clone()
    apply(node, [selection, movement])
    after = live_positions(node, color)
    return after if after != before else ()


def report(root, name, path):
    node = root.clone()
    try:
        apply(node, path)
    except Exception as exc:
        print(name, "ERROR", type(exc).__name__, str(exc))
        return
    print(
        name,
        "n", len(path),
        "L", node.levels_completed,
        "H", head_positions(node),
        "8", live_positions(node, 8),
        "9", live_positions(node, 9),
        "own8", follows(node, 8, (6, 32, 5), 3),
        "own9", follows(node, 9, (6, 8, 29), 1),
    )


def probe(env):
    reach_level_6(env)
    top = (6, 32, 5)
    for retract in range(7):
        report(env, f"h_e6_r{retract}", [4] * 6 + [3] * retract)
    for retract in range(7):
        report(
            env,
            f"v_e6_r{retract}",
            [top] + [2] * 6 + [1] * retract,
        )
    report(
        env,
        "both_e6_r6",
        [4] * 6 + [3] * 6 + [top] + [2] * 6 + [1] * 6,
    )
    report(env, "first8", FIRST_EIGHT)
    report(env, "first8_transfer", FIRST_EIGHT + ["v", 2, 1])
    report(env, "first8_coord7", FIRST_EIGHT + [(7, 32, 10)])
    report(
        env,
        "first8_select_coord7",
        FIRST_EIGHT + ["v", (7, 32, 10)],
    )
    report(
        env,
        "second8_stage",
        FIRST_EIGHT + ["v", 2, 1, "h"] + NEXT_EIGHT,
    )
    report(env, "first9", ["v"] + FIRST_NINE)
    report(
        env,
        "first9_coord7",
        ["v"] + FIRST_NINE + [(7, 14, 28)],
    )
    report(
        env,
        "first9_transfer",
        ["v"] + FIRST_NINE + ["h", 4, "v", 1, 1, 1],
    )
    report(
        env,
        "second9_stage",
        ["v"] + FIRST_NINE + ["h", 4, "v", 1, 1, 1]
        + ["v"] + NEXT_NINE,
    )
    second8 = (
        FIRST_EIGHT + ["v", 2, 1, "h"] + NEXT_EIGHT
        + ["v", 2, "h", 3, 3, 3]
    )
    third8 = second8 + ["h"] + NEXT_EIGHT + ["v", 2, "h", 3, 3, 3]
    parked8 = third8 + ["v", 1, 1, 1]
    report(env, "second8_kept", second8)
    report(env, "third8_kept", third8)
    report(env, "parked8", parked8)

    clean9_1 = ["v"] + FIRST_NINE
    clean9_2 = clean9_1 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    clean9_3 = clean9_2 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    report(env, "clean9_2", clean9_2)
    report(env, "clean9_3", clean9_3)
    loaded9_home = clean9_3 + ["v", 4, "h"]
    report(env, "loaded9_down1", loaded9_home + [2])
    report(env, "loaded9_down1_up4", loaded9_home + [2] + [1] * 4)
    report(
        env,
        "loaded9_down1_cycle_up4",
        loaded9_home + [2, 4, 3] + [1] * 4,
    )
    report(
        env,
        "loaded9_down1_extend_up4",
        loaded9_home + [2, 4] + [1] * 4,
    )

    combined9_1 = parked8 + FIRST_NINE
    combined9_2 = combined9_1 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    combined9_3 = combined9_2 + NEXT_NINE + ["h", 4, "v", 1, 1, 1]
    report(env, "combined9_1", combined9_1)
    report(env, "combined9_2", combined9_2)
    report(env, "combined9_3", combined9_3)
    for suffix in (["v", 4], ["v", 4, 2], ["v", 4, 2, 2],
                   ["v", 4, 2, 2, 2]):
        report(env, "combined_return_" + "".join(map(str, suffix[1:])),
               combined9_3 + suffix)

    carry9_1 = third8 + ["v"] + FIRST_NINE
    carry9_2 = carry9_1 + ["v"] + NEXT_NINE + [
        "h", 4, "v", 1, 1, 1
    ]
    carry9_3 = carry9_2 + ["v"] + NEXT_NINE + [
        "h", 4, "v", 1, 1, 1
    ]
    report(env, "carry9_1", carry9_1)
    report(env, "carry9_2", carry9_2)
    report(env, "carry9_3", carry9_3)
    report(env, "carry9_home_vh", carry9_3 + ["v", 4, "h", 2])
    report(env, "carry9_home_hv", carry9_3 + ["h", 2, "v", 4])

    carry8_1 = clean9_3 + ["v", 4, "h"] + FIRST_EIGHT
    carry8_2 = carry8_1 + ["v", 2, 1, "h"] + NEXT_EIGHT + [
        "v", 2, "h", 3, 3, 3
    ]
    carry8_3 = carry8_2 + ["h"] + NEXT_EIGHT + [
        "v", 2, "h", 3, 3, 3
    ]
    report(env, "carry8_1", carry8_1)
    report(env, "carry8_2", carry8_2)
    report(env, "carry8_3", carry8_3)
    report(env, "carry8_home_hv", carry8_3 + ["h", 2, "v", 4])
    report(env, "carry8_home_vh", carry8_3 + ["v", 4, "h", 2])


A.run_program("sk48", probe)
