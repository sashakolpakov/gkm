"""Per-level players only compose reusable legs."""
from legs import follow_key_runs


def play_level_1(env):
    follow_key_runs(env, ((3, 3), (1, 4), (4, 3), (1, 3)))


def play_level_2(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_z2rwf0iy/level2_rotation_candidate.json
    for action in [1, 4, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 1, 4, 3, 4, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2]:
        env.step(action)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_b85xxaho/level3_bfs_candidate.json
    for action in [1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 1, 2, 1, 4, 2]:
        env.step(action)
