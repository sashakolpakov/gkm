from legs import repeat_key, enter_displayed_pattern, activate_palette_pattern


def play_level_1(env):
    repeat_key(env, 3, 5)
    enter_displayed_pattern(env)
    repeat_key(env, 3, 4)


def play_level_2(env):
    enter_displayed_pattern(env, ink=11)
    repeat_key(env, 1, 2)


def play_level_3(env):
    repeat_key(env, 4, 2)
    enter_displayed_pattern(env, ink=6)
    repeat_key(env, 3, 4)
    repeat_key(env, 2, 4)
    repeat_key(env, 3, 1)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sc25_arc_agi3_n0_fresh_frontier_uize5e8g/level4_palette_ray_candidate.json
    for action in [2, 2, [6, 4, 4], [6, 30, 50], [6, 25, 55], [6, 35, 55], [6, 30, 60], 2, 3, [6, 4, 15], [6, 30, 50], [6, 30, 55], [6, 30, 60], 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4]:
        env.step(action)
