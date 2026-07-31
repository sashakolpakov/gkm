import players

def solve(env):
    if env.levels_completed == 0:
        players.play_level_1(env)
