# Per-level players only compose reusable legs.
from legs import reunite_mirrored_pair, park_selectable_blockers


def play_level_1(env):
    reunite_mirrored_pair(env)


def play_level_2(env):
    reunite_mirrored_pair(env, hazard_colors=(8,))


def play_level_3(env):
    park_selectable_blockers(env)
    reunite_mirrored_pair(env, hazard_colors=(8,))


def play_level_4(env):
    park_selectable_blockers(env, allow_chamber_parking=True)
    reunite_mirrored_pair(env, hazard_colors=(8,))


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_m0r0_arc_agi3_n0_fresh_frontier_ghm_dkt0/pressure_reunion_level5_candidate.json
    for action in [3, 3, 1, 1, 1, 1, 3, 1, 4, 4, 1, 4, 4, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 4, 4, 1, 3, 1, 1, 1, 1, 3, 3, 3]:
        env.step(action)
