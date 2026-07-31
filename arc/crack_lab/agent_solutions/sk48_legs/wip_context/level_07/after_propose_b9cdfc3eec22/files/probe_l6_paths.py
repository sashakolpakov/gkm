import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def token_positions(env):
    blobs = connected_components(env.frame(), colors=(8, 9), min_area=4)
    return {
        color: sorted(
            (round(blob.centroid[0]), round(blob.centroid[1]), blob.area)
            for blob in blobs if blob.color == color
        )
        for color in (8, 9)
    }


def run_path(root, name, path):
    node = root.clone()
    snapshots = []
    for index, action in enumerate(path, 1):
        node.step(*action) if isinstance(action, tuple) else node.step(action)
        positions = token_positions(node)
        if not snapshots or positions != snapshots[-1][1] or node.levels_completed > 5:
            snapshots.append((index, positions, node.levels_completed))
    print(name, "len", len(path), "snapshots", snapshots)


def run_final(root, name, path):
    node = root.clone()
    try:
        for action in path:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
    except Exception as exc:
        print("FINAL", name, "ERROR", type(exc).__name__, str(exc))
        return
    pieces = connected_components(node.frame(), colors=(1, 8, 9), min_area=4)
    compact = [(b.color, tuple(round(x) for x in b.centroid), b.area) for b in pieces]
    print("FINAL", name, "L", node.levels_completed, "pieces", compact)


def probe(env):
    reach_level_6(env)
    first = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
    second = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 5 + [4] + [3] * 3
    third = [2] * 6 + [4] * 6 + [3] * 4 + [1] * 6 + [4] + [3] * 3
    transfer_to_top = [(6, 32, 5), 2, 1]
    next_eight = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
    route_eights = (
        first + transfer_to_top
        + [(6, 8, 10)] + next_eight + transfer_to_top
        + [(6, 8, 16)] + [3] * 3 + next_eight + transfer_to_top
    )
    first_nine = (
        [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
    )
    transfer_to_left = [(6, 8, 29), 4, 3]
    next_nine = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]
    route_nines = (
        [(6, 32, 5)] + first_nine + transfer_to_left
        + [(6, 14, 5)] + next_nine + transfer_to_left
        + [(6, 20, 5)] + [1] * 3 + next_nine + transfer_to_left
    )
    stage_second_16 = (
        first + transfer_to_top + [(6, 8, 10)]
        + [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
    )
    interleaved_second_16 = (
        stage_second_16
        + [(6, 32, 5), 2, (6, 8, 16)] + [3] * 3
        + [(6, 32, 5), 1]
    )
    detached_second_16 = (
        stage_second_16
        + [(6, 32, 5), 2, (6, 8, 16)] + [3] * 3
    )
    stage_third_22 = (
        interleaved_second_16 + [(6, 8, 16)]
        + [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
    )
    shifted_third = (
        interleaved_second_16 + [3] + [(6, 8, 16)]
        + [2] * 5 + [4] * 6 + [3] * 4 + [1] * 3
        + [(6, 26, 5), 2, (6, 8, 28)] + [3] * 2
        + [(6, 26, 5), 1, 4]
    )
    shifted_stage = (
        interleaved_second_16 + [3] + [(6, 8, 16)]
        + [2] * 5 + [4] * 6 + [3] * 4 + [1] * 3
    )
    interleaved_second_nine = (
        [(6, 32, 5)] + first_nine + transfer_to_left
        + [(6, 14, 5)] + next_nine
        + [(6, 8, 29), 4, (6, 20, 5)] + [1] * 3
        + [(6, 8, 29), 3]
    )
    stage_third_nine = (
        interleaved_second_nine + [(6, 20, 5)]
        + [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]
    )
    stage_second_22 = (
        first + transfer_to_top + [(6, 8, 10)]
        + [2] * 5 + [4] * 6 + [3] * 4 + [1] * 3 + [4]
    )
    cases = {
        "left_extend_3_to_8": [4] * 8,
        "left_extend6_retract0to6": [4] * 6 + [3] * 6,
        "top_extend3_to_8": [(6, 32, 5)] + [2] * 8,
        "top_extend6_retract0to6": [(6, 32, 5)] + [2] * 6 + [1] * 6,
        "left_then_top": (
            [4] * 6 + [3] * 3
            + [(6, 32, 5)] + [2] * 6 + [1] * 3
        ),
        "top_then_left": (
            [(6, 32, 5)] + [2] * 6 + [1] * 3
            + [(6, 8, 29)] + [4] * 6 + [3] * 3
        ),
        "inherited_first": first,
        "inherited_second": second,
        "inherited_third": third,
        "inherited_all": first + second + third,
        "route_eights": route_eights,
        "route_nines": route_nines,
        "route_both": route_eights + route_nines,
    }
    print("initial", token_positions(env))
    for name, path in cases.items():
        run_path(env, name, path)
    for name, suffix in {
        "stage_click": [(6, 32, 5)],
        "stage_click_up": [(6, 32, 5), 1],
        "stage_click_down": [(6, 32, 5), 2],
        "stage_click_up2": [(6, 32, 5), 1, 1],
        "stage_click_down_up": [(6, 32, 5), 2, 1],
        "stage_click_token": [(6, 32, 10)],
        "stage_key7": [7],
        "stage_coord7_token": [(7, 32, 10)],
        "stage_select_release": [(6, 32, 5), 7],
        "stage_select_coord7": [(6, 32, 5), (7, 32, 10)],
    }.items():
        run_final(env, name, first + suffix)
    run_final(env, "route_eights", route_eights)
    run_final(env, "route_nines", route_nines)
    run_final(env, "route_both", route_eights + route_nines)
    run_final(env, "stage_second_16", stage_second_16)
    run_final(env, "interleaved_second_16", interleaved_second_16)
    run_final(env, "detached_second_16", detached_second_16)
    run_final(
        env,
        "detached_top_up2",
        detached_second_16 + [(6, 32, 5), 1, 1],
    )
    run_final(
        env,
        "interleaved_click_second_left",
        interleaved_second_16 + [(6, 32, 16), 3],
    )
    run_final(
        env,
        "interleaved_cycle_click_second_left",
        interleaved_second_16 + [2, 1, (6, 32, 16), 3],
    )
    run_final(env, "stage_third_22", stage_third_22)
    run_final(env, "third_direct_retract", stage_third_22 + [3] * 3)
    run_final(
        env,
        "third_interleaved",
        stage_third_22
        + [(6, 32, 5), 2, (6, 8, 22)] + [3] * 3
        + [(6, 32, 5), 1],
    )
    run_final(env, "shifted_third", shifted_third)
    run_final(env, "shifted_stage", shifted_stage)
    run_final(env, "shifted_top_down", shifted_stage + [(6, 26, 5), 2])
    run_final(env, "shifted_top_down2", shifted_stage + [(6, 26, 5), 2, 2])
    run_final(
        env,
        "shifted_down2_h_retract",
        shifted_stage + [(6, 26, 5), 2, 2, (6, 8, 28), 3, 3],
    )
    run_final(env, "interleaved_second_nine", interleaved_second_nine)
    run_final(
        env,
        "second_nine_extra_retract_move",
        interleaved_second_nine + [3, 1],
    )
    run_final(
        env,
        "second_nine_cycle_move",
        interleaved_second_nine + [4, 3, 1],
    )
    run_final(
        env,
        "second_nine_cycle_retract_move",
        interleaved_second_nine + [4, 3, 3, 1],
    )
    run_final(
        env,
        "second_eight_extra_retract_move",
        interleaved_second_16 + [1, 3],
    )
    run_final(
        env,
        "second_eight_cycle_move",
        interleaved_second_16 + [2, 1, 3],
    )
    run_final(env, "stage_third_nine", stage_third_nine)
    run_final(
        env,
        "third_nine_left_up",
        stage_third_nine + [(6, 8, 29), 1],
    )
    run_final(
        env,
        "third_nine_clear_lane",
        stage_third_nine + [(6, 8, 29), 1, (6, 26, 5), 1],
    )
    run_final(
        env,
        "third_nine_join_return",
        stage_third_nine
        + [(6, 8, 29), 1, (6, 26, 5), 1]
        + [(6, 8, 23), 4, 3, 2],
    )
    run_final(env, "eights_then_nines", stage_third_22 + stage_third_nine)
    run_final(env, "nines_then_eights", stage_third_nine + stage_third_22)
    run_final(env, "stage_second_22_top_down_up",
              stage_second_22 + [(6, 32, 5), 2, 1])


A.run_program("sk48", probe)
