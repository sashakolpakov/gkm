# Each level player composes reusable legs.
from legs import (click_walk_to_region, merge_matching_markers,
                  merge_preserving_mass)


def play_level_1(env):
    click_walk_to_region(env)


def play_level_2(env):
    merge_matching_markers(env)
    click_walk_to_region(env, marker_color=11, stride=3)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_arc_agi3_n0_fresh_frontier_rknmzzuf/candidate_two_goal_merge.json
    for action in [[6, 58, 23], [6, 10, 25], [6, 31, 18], [6, 50, 23], [6, 22, 17], [6, 13, 26], [6, 16, 27], [6, 19, 28], [6, 22, 29], [6, 25, 30], [6, 24, 21], [6, 48, 27], [6, 46, 31], [6, 44, 35], [6, 42, 39], [6, 40, 43], [6, 38, 47], [6, 36, 50], [6, 34, 50], [6, 32, 50], [6, 30, 50], [6, 28, 50], [6, 26, 50], [6, 24, 50], [6, 22, 50], [6, 20, 50], [6, 18, 50], [6, 16, 50], [6, 14, 50], [6, 12, 50], [6, 10, 50], [6, 9, 50], [6, 9, 50], [6, 23, 25], [6, 23, 29], [6, 23, 33], [6, 23, 37], [6, 23, 41], [6, 23, 45], [6, 23, 49]]:
        env.step(action)


def play_level_4(env):
    merge_preserving_mass(env)
    click_walk_to_region(env, marker_color=11, stride=3, max_clicks=40)
