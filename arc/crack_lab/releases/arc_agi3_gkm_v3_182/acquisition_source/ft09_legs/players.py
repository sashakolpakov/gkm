# Per-level players only compose reusable legs.
from legs import match_miniature, match_coupled_miniatures


def play_level_1(env):
    match_miniature(env, 36, 36)


def play_level_2(env):
    match_miniature(env, 20, 14, additional_boards=((20, 30),))


def play_level_3(env):
    match_miniature(env, 20, 4, tile_colors=(8, 12),
                    additional_boards=((12, 20), (28, 20), (20, 36)))


def play_level_4(env):
    match_miniature(env, 12, 14, tile_colors=(9, 8, 12),
                    additional_boards=((28, 14), (20, 30)))


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ft09_arc_agi3_n0_fresh_frontier_tspxtzr0/coupled_miniature_candidate.json
    for action in [[6, 25, 7], [6, 17, 15], [6, 25, 15], [6, 33, 7], [6, 33, 15], [6, 33, 23], [6, 49, 23], [6, 17, 23], [6, 17, 31], [6, 17, 39], [6, 33, 31], [6, 33, 39], [6, 41, 39], [6, 49, 39], [6, 25, 31], [6, 33, 47], [6, 17, 55], [6, 33, 55], [6, 41, 47], [6, 49, 47], [6, 41, 55]]:
        env.step(action)


def play_level_6(env):
    match_coupled_miniatures(env, ((12, 6), (36, 22), (20, 30), (44, 46)),
                             tile_colors=(11, 14))
