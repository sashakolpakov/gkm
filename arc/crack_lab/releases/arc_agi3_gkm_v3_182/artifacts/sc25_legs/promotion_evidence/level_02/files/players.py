from legs import repeat_key, enter_displayed_pattern


def play_level_1(env):
    repeat_key(env, 3, 5)
    enter_displayed_pattern(env)
    repeat_key(env, 3, 4)


def play_level_2(env):
    enter_displayed_pattern(env, ink=11)
    repeat_key(env, 1, 2)
