from legs import fill_marked_cells, handoff_across_barrier


def play_level_1(env):
    fill_marked_cells(env)


def play_level_2(env):
    fill_marked_cells(env, cooperative=True)


def play_level_3(env):
    handoff_across_barrier(env)
