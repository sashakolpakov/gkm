# Each player composes reusable legs only.
from legs import (align_horizontal_indicator, align_indicator_with_transfers,
                  align_reservoir_chain, float_through_locks)


def play_level_1(env):
    align_horizontal_indicator(env)


def play_level_2(env):
    align_indicator_with_transfers(env)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_ujd0or1x/level3_transfer_candidate.json
    for action in [[6, 12, 56], [6, 24, 56], [6, 12, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 34, 56], [6, 24, 56], [6, 12, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56], [6, 46, 56]]:
        env.step(action)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_arc_agi3_n0_fresh_frontier_bh2kb6jb/level4_float_route_candidate.json
    for action in [[6, 16, 62], [6, 16, 62], [6, 13, 48], [6, 16, 62], [6, 16, 62], [6, 16, 62], [6, 52, 62], [6, 52, 62], [6, 52, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 28, 39], [6, 52, 62], [6, 52, 62], [6, 52, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62], [6, 40, 62]]:
        env.step(action)
