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


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_6ah5roa1/level4_staged_candidate.json
    for action in [3, 3, 3, 2, 2, 2, 4, 1, 2, 3, 3, 1, 1, 1, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 4, 4, 2, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 4, 2, 2, 3, 3, 3, 1, 1, 3, 3, 3]:
        env.step(action)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_owwcfm9h/level5_pattern_rotation_candidate.json
    for action in [1, 3, 1, 1, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 2, 2, 3, 3, 3, 1, 3, 4, 3, 3, 3, 4, 2, 4, 2, 2, 2, 2, 4, 4, 3, 2, 2, 4, 4, 2, 4, 4, 4, 4, 4, 4, 1]:
        env.step(action)


def play_level_6(env):
    # Verified ring collection, pattern changes, and both matching panels.
    follow_key_runs(env, ((1, 1), (3, 1), (1, 1), (3, 2), (1, 3), (4, 2), (3, 2), (1, 4), (4, 6), (2, 2), (4, 2), (1, 1), (4, 1), (1, 1), (3, 1), (2, 4), (4, 1), (2, 2), (1, 2), (3, 1), (1, 3), (3, 4), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 3), (2, 7), (1, 2), (2, 1), (4, 3), (3, 3), (1, 3), (4, 2), (1, 1), (2, 1), (3, 2), (1, 4), (4, 6), (2, 2), (4, 2), (1, 1), (4, 1), (1, 1), (3, 1), (4, 1), (2, 5)))
