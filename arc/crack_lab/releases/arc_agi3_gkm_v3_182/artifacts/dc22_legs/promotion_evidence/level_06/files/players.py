# Per-level players only compose reusable legs.
from legs import navigate_switchable_platforms, manipulate_platform_controls


def play_level_1(env):
    navigate_switchable_platforms(env, switches=((48, 20), (48, 36)))


def play_level_2(env):
    navigate_switchable_platforms(env, switches=((52, 22), (52, 40)),
                                  waypoint=(54, 18), max_depth=32)
    navigate_switchable_platforms(env,
                                  switches=((52, 22), (52, 30), (52, 40)))


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_47p03idf/candidate_level3_branch_pickup.json
    for action in [[6, 52, 27], [6, 52, 18], 3, 3, 3, 3, 3, 2, 2, 2, [6, 52, 18], [6, 52, 27], 3, 3, 3, 3, 3, 3, 3, [6, 52, 27], 1, 1, 1, 1, [6, 52, 36], 4, 1, 1, 4, [6, 52, 45], 1, 1, 1, 1, 1, [6, 52, 18], 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2]:
        env.step(action)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_f99uwy6u/candidate_level4_staged.json
    for action in [2, 4, [6, 56, 28], 4, [6, 56, 28], 4, [6, 56, 28], 4, [6, 56, 28], 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, [6, 45, 28], [6, 45, 28], [6, 45, 28], [6, 45, 28], [6, 45, 28], 3, 3, 3, 2, 2, [6, 52, 20], 2, 4, 4, [6, 45, 28], 2, 1, 1, 3, 3, [6, 52, 20], 1, [6, 45, 28], [6, 45, 28], [6, 45, 28], 1, 2, 2, [6, 52, 20], 2, 4, [6, 45, 28], [6, 45, 28], [6, 45, 28], 4, 2, 2, 3, 2, 2, 2, 2, 4, [6, 45, 28], 4, [6, 45, 28], 4, [6, 45, 28], 4, [6, 45, 28], 4, [6, 56, 28], 2, [6, 56, 28], 4, [6, 56, 28], 4, [6, 56, 28], 4, 4, 4, 4, 4, 2, 2]:
        env.step(action)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier__b4isum0/candidate_level5_win.json
    for action in [[6, 50, 29], [6, 50, 29], [6, 50, 29], [6, 55, 29], [6, 55, 29], [6, 55, 29], [6, 52, 35], [6, 45, 29], [6, 45, 29], [6, 45, 29], [6, 60, 29], [6, 60, 29], [6, 60, 29], [6, 55, 29], [6, 55, 29], [6, 55, 29], 3, [6, 52, 41], 3, 1, 1, 3, 3, 1, 1, 1, 1, 4, 1, [6, 45, 29], [6, 45, 29], [6, 45, 29], [6, 50, 29], [6, 50, 29], [6, 50, 29], [6, 55, 29], [6, 55, 29], [6, 55, 29], 3, 1, 1, 1, 1, 1, 1, 1, 3, 4, [6, 52, 48], [6, 45, 29], [6, 45, 29], [6, 45, 29], [6, 60, 29], [6, 60, 29], [6, 60, 29], 1, 1, 1, 1, 1, 1, 1, 1, [6, 52, 14], [6, 56, 22], [6, 56, 22], [6, 56, 22], [6, 56, 22], [6, 47, 22], [6, 47, 22], [6, 47, 22], [6, 47, 22], 2, [6, 52, 41], 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, [6, 52, 14], 1, 1, 3, 3, 3, 3, 3, 3]:
        env.step(action)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_yveteuv7/candidate_level6_verified_win.json
    for action in [3, 3, 3, 3, 3, 1, 1, [6, 51, 25], 4, 1, 1, [6, 54, 6], [6, 51, 18], 2, 2, 3, [6, 51, 25], 2, 2, [6, 54, 6], [6, 54, 6], [6, 54, 6], 2, 3, 3, 2, 3, 2, 3, [6, 54, 6], 1, 3, [6, 54, 6], 1, 1, 1, 1, 1, 1, [6, 51, 25], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, [6, 51, 25], 4, 4, 4, 2, 3, 3, 3, 3, 2, [6, 51, 25], 2, 2, 2, 2, 2, 2, [6, 54, 6], 2, 2, 2, 4, [6, 54, 6], 4, [6, 54, 6], 4, [6, 54, 6], 4, 1, 1, 1, [6, 50, 46], [6, 50, 46], [6, 50, 46], [6, 51, 25], 1, 2, 3, [6, 47, 37], [6, 51, 18], [6, 47, 37], 4, 2, [6, 51, 41], [6, 51, 41], [6, 51, 41], [6, 51, 41], [6, 51, 41], [6, 51, 41], [6, 51, 18], 1, 3, [6, 47, 37], [6, 47, 37], [6, 51, 18], 4, 4, [6, 55, 37], [6, 55, 37], [6, 55, 37], [6, 55, 37], [6, 55, 37], 3, 1, [6, 51, 33], [6, 51, 33], [6, 51, 33], 2, 3, [6, 47, 37], [6, 47, 37], 4, 1, [6, 51, 33], [6, 51, 33], [6, 51, 33], 2, [6, 51, 25], [6, 50, 46], [6, 50, 46], [6, 50, 46], [6, 51, 25], 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4]:
        env.step(action)
