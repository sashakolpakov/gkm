from legs import fill_marked_cells, handoff_across_barrier, deliver_with_moving_carrier, clear_interfering_carrier, yield_until_complete


def play_level_1(env):
    fill_marked_cells(env)


def play_level_2(env):
    fill_marked_cells(env, cooperative=True)


def play_level_3(env):
    handoff_across_barrier(env)


def play_level_4(env):
    handoff_across_barrier(env)


def play_level_5(env):
    deliver_with_moving_carrier(env)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_wa30_arc_agi3_n0_fresh_frontier_19jl4di6/outlined_delivery_candidate.json
    for action in [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 5, 4, 4, 5, 3, 3, 3, 1, 1, 3, 3, 5, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 5, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3, 5]:
        env.step(action)


def play_level_7(env):
    clear_interfering_carrier(env, anticipate_motion=True)
    fill_marked_cells(env, target_signature=(2, 9), persistent_targets=True)


def play_level_8(env):
    clear_interfering_carrier(env, anticipate_motion=True)
    fill_marked_cells(env, target_signature=(2, 9), persistent_targets=True,
                      pickup_directions=(2,),
                      delivery_pairs=[((2, 2), (2, 14)), ((2, 1), (2, 13))])
    yield_until_complete(env)
