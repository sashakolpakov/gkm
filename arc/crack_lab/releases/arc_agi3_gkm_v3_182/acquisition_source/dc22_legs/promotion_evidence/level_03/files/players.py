# Per-level players only compose reusable legs.
from legs import navigate_switchable_platforms


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
