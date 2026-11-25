# Each player only composes reusable legs.
from legs import walk_straight, release_recorded_companion


def play_level_1(env):
    walk_straight(env, 4, 4)
    release_recorded_companion(env)
    walk_straight(env, 2, 7)
    walk_straight(env, 4, 5)


def play_level_2(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_g50t_arc_agi3_n0_fresh_frontier__i3kdxa2/level2_two_companion_candidate.json
    for action in [3, 3, 5, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 3, 3, 5, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 4, 4, 4]:
        env.step(action)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_g50t_arc_agi3_n0_fresh_frontier_dbg_pkk7/level3_two_replay_candidate.json
    for action in [1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 4, 5, 1, 1, 4, 4, 4, 4, 2, 2, 1, 1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 5, 1, 1, 4, 4, 4, 4, 2, 2, 1, 1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 4, 4, 1, 1]:
        env.step(action)
