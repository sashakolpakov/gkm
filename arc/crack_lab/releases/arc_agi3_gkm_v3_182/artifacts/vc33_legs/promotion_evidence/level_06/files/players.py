# Each player composes reusable legs only.
from legs import (align_horizontal_indicator, align_indicator_with_transfers,
                  align_reservoir_chain, float_through_locks,
                  exchange_figures_through_locks, route_through_shared_reservoir)


def play_level_1(env):
    align_horizontal_indicator(env)


def play_level_2(env):
    align_indicator_with_transfers(env)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_k8b84wqt/level6_branch_route_candidate.json
    for action in [[6, 1, 28], [6, 25, 28], [6, 25, 28], [6, 25, 28], [6, 11, 31], [6, 1, 34], [6, 1, 34], [6, 25, 34], [6, 25, 34], [6, 25, 34], [6, 25, 34], [6, 25, 34], [6, 25, 34], [6, 35, 31], [6, 25, 28], [6, 25, 28], [6, 25, 28], [6, 25, 28], [6, 25, 28], [6, 25, 28]]:
        env.step(action)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_ujd0or1x/level3_transfer_candidate.json
    for action in [[6, 12, 56], [6, 24, 56], [6, 12, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56]]:
        env.step(action)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_bh2kb6jb/level4_float_route_candidate.json
    for action in [[6, 16, 62], [6, 16, 62], [6, 13, 48], [6, 16, 62], [6, 16, 62], [6, 16, 62], [6, 52, 62], [6, 52, 62], [6, 52, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 28, 39], [6, 52, 62], [6, 52, 62], [6, 52, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62]]:
        env.step(action)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_nameengg/level5_counterflow_candidate.json
    for action in [[6, 62, 18], [6, 62, 18], [6, 62, 18], [6, 62, 18], [6, 62, 36], [6, 62, 36], [6, 62, 36], [6, 62, 36], [6, 62, 36], [6, 62, 53], [6, 62, 53], [6, 30, 50], [6, 62, 30], [6, 62, 30], [6, 62, 30], [6, 62, 53], [6, 62, 53], [6, 46, 33], [6, 62, 18], [6, 62, 18], [6, 62, 18], [6, 62, 18], [6, 34, 15], [6, 62, 12], [6, 62, 12], [6, 62, 12], [6, 62, 12], [6, 46, 33], [6, 62, 12], [6, 62, 36], [6, 62, 36], [6, 62, 36], [6, 62, 47], [6, 62, 47], [6, 30, 50], [6, 62, 30], [6, 62, 12], [6, 62, 53], [6, 62, 53], [6, 62, 53], [6, 62, 53], [6, 62, 53], [6, 62, 53], [6, 62, 53]]:
        env.step(action)
