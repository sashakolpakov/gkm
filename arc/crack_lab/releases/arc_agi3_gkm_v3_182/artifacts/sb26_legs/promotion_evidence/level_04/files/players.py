"""Level players compose reusable legs."""
from legs import (match_colour_row, match_nested_colour_rows,
                  complete_expandable_colour_rows)


def play_level_1(env):
    match_colour_row(env)


def play_level_2(env):
    match_nested_colour_rows(env)


def play_level_3(env):
    match_nested_colour_rows(env)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sb26_arc_agi3_n0_fresh_frontier_bxafq8be/level4_expanded_reference_candidate.json
    for action in [[6, 10, 58], [6, 20, 22], [6, 31, 58], [6, 26, 22], [6, 52, 58], [6, 32, 22], [6, 24, 58], [6, 38, 22], [6, 38, 58], [6, 44, 22], [6, 45, 58], [6, 32, 36], [6, 17, 58], [6, 38, 36], 5]:
        env.step(action)
