from legs import collect_ordered_chain, stage_chain_blockers, clear_chain_guide, distribute_chain_to_rails, relay_chain_around_barrier


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
