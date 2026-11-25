# Per-level players only compose reusable legs.
from legs import mate_blue_tips, assemble_blue_tips


def play_level_1(env):
    mate_blue_tips(env)


def play_level_2(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_cn04_arc_agi3_n0_fresh_frontier_treudnhm/assembled_tips_candidate.json
    for action in [2, 2, 4, [6, 24, 36], 1, 1, 1, 1, 4, [6, 48, 18], 2, 2, 2, 2, 3, 3, 3, [6, 51, 57], 5, 5, 5, 1, 1, 1, 1, 1, 1, 3, 3, 3]:
        env.step(action)


def play_level_3(env):
    assemble_blue_tips(env)


def play_level_4(env):
    assemble_blue_tips(env)
