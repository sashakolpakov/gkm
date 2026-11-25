from legs import collect_ordered_chain, stage_chain_blockers, clear_chain_guide, distribute_chain_to_rails, relay_chain_around_barrier, exchange_cargo_between_shafts


def play_level_1(env):
    collect_ordered_chain(env)


def play_level_2(env):
    stage_chain_blockers(env)
    collect_ordered_chain(env)


def play_level_3(env):
    collect_ordered_chain(env, clearance=clear_chain_guide)


def play_level_4(env):
    distribute_chain_to_rails(env)


def play_level_5(env):
    relay_chain_around_barrier(env)
    collect_ordered_chain(env, lowest_first=True)


def play_level_6(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sk48_arc_agi3_n0_fresh_frontier_3xfl_99l/cooperative_gripper_candidate.json
    for action in [2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, [6, 32, 5], 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 1, 1, 1, 1, 4, 2, 2, 2, 2, 1, 1, 1, 1, 4, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, [6, 8, 29], 2, 4, 4, 4, 4, 4, 4, 4]:
        env.step(action)
