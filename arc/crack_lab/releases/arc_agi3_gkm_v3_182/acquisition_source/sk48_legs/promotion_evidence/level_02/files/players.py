from legs import collect_ordered_chain, stage_chain_blockers


def play_level_1(env):
    collect_ordered_chain(env)


def play_level_2(env):
    stage_chain_blockers(env)
    collect_ordered_chain(env)
