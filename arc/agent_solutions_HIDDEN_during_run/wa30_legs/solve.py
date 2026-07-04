import players

def solve(env):
    # dispatch to the per-level player for the current level, in a loop
    while not env.terminal():
        k = env.levels_completed + 1
        fn = getattr(players, f'play_level_{k}', None)
        if fn is None:
            return
        before = env.levels_completed
        fn(env)
        if env.levels_completed <= before:
            return  # no progress -> stop
