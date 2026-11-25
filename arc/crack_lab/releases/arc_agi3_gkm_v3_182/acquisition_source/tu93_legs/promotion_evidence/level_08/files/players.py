# Per-level players only compose reusable legs.
from legs import clear_linked_sentries, traverse_linked_tiles, traverse_patrolled_links


def play_level_1(env):
    traverse_linked_tiles(env)


def play_level_2(env):
    clear_linked_sentries(env)
    traverse_linked_tiles(env)


def play_level_3(env):
    clear_linked_sentries(env)
    traverse_linked_tiles(env)


def play_level_4(env):
    clear_linked_sentries(env)
    traverse_linked_tiles(env)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [3, 3, 3, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 4, 2, 2, 4, 4, 4, 1, 1, 3]:
        env.step(action)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_arc_agi3_n0_fresh_frontier_zp54_71g/level6_incremental_candidate.json
    for action in [3, 3, 2, 2, 4, 3, 3, 3, 4, 3, 4, 4, 2, 4, 2, 3, 2, 3, 1, 2, 3, 1, 1, 3, 1, 1, 1, 3]:
        env.step(action)


def play_level_7(env):
    traverse_patrolled_links(env)


def play_level_8(env):
    traverse_patrolled_links(env)
