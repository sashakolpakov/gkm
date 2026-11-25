# Per-level players only compose reusable legs.
from legs import (align_marked_bar_tips, route_telescoping_arm_through_gap,
                  park_crossing_telescopic_barriers, stage_rotating_arm_through_gap,
                  stage_joint_controls)


def play_level_1(env):
    align_marked_bar_tips(env)


def play_level_2(env):
    route_telescoping_arm_through_gap(env)


def play_level_3(env):
    park_crossing_telescopic_barriers(env)
    align_marked_bar_tips(env)


def play_level_4(env):
    park_crossing_telescopic_barriers(
        env, stages=((14, 1, 4), (9, 1, 4), (12, 0, 7), (8, 1, 4)))
    align_marked_bar_tips(env)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_arc_agi3_n0_fresh_frontier_nxg8j0d5/candidate_coupled_parking_level5.json
    for action in [[6, 52, 57], [6, 52, 57], [6, 52, 57], [6, 52, 57], [6, 52, 57], [6, 26, 57], [6, 26, 57], [6, 26, 57], [6, 41, 57], [6, 41, 57], [6, 41, 57], [6, 41, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 7, 57], [6, 37, 57], [6, 37, 57], [6, 37, 57], [6, 37, 57], [6, 26, 57], [6, 26, 57], [6, 26, 57], [6, 26, 57], [6, 26, 57], [6, 26, 57], [6, 41, 57], [6, 41, 57], [6, 41, 57], [6, 41, 57], [6, 56, 57], [6, 56, 57], [6, 56, 57], [6, 56, 57], [6, 56, 57]]:
        env.step(action)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_arc_agi3_n0_fresh_frontier_hiku9hh9/candidate_rotating_arm_level6.json
    for action in [[6, 12, 48], [6, 12, 48], [6, 50, 48], [6, 31, 48], [6, 31, 48], [6, 53, 57], [6, 53, 57], [6, 53, 57], [6, 53, 57], [6, 53, 57], [6, 53, 57], [6, 53, 57], [6, 34, 57], [6, 34, 57], [6, 34, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57], [6, 15, 57]]:
        env.step(action)


def play_level_7(env):
    stage_joint_controls(env, stages=(
        (10, 0, 3), (8, 0, 2), (11, 2, 2), (11, 1, 8),
        (9, 0, 1), (9, 2, 1), (14, 1, 12), (9, 2, 2), (9, 1, 7)))
    align_marked_bar_tips(env)


def play_level_8(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_arc_agi3_n0_fresh_frontier_gnb5evw9/candidate_level8_rotating_cross.json
    for action in [[6, 49, 18], [6, 49, 18], [6, 49, 18], [6, 44, 4], [6, 44, 4], [6, 44, 4], [6, 23, 56], [6, 23, 56], [6, 23, 56], [6, 23, 56], [6, 23, 56], [6, 23, 56], [6, 40, 4], [6, 40, 4], [6, 40, 4], [6, 49, 18], [6, 49, 18], [6, 49, 18], [6, 44, 11], [6, 44, 11], [6, 44, 11], [6, 9, 56], [6, 9, 56], [6, 9, 56], [6, 9, 56], [6, 9, 56], [6, 40, 11], [6, 40, 11], [6, 40, 11], [6, 49, 18], [6, 49, 18], [6, 49, 18], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11], [6, 58, 11]]:
        env.step(action)
