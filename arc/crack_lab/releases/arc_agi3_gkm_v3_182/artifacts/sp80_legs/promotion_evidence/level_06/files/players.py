# Per-level players only compose reusable legs.
from legs import align_horizontal, align_vertical, activate, select_object


def play_level_1(env):
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-4)
    activate(env)


def play_level_2(env):
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-8)
    select_object(env, color=8)
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-16)
    select_object(env, color=8)
    align_horizontal(env, moving_color=9, reference_color=6)
    activate(env)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [[6, 9, 21], 4, [6, 41, 29], 3, 3, [6, 9, 33], 3, 3, [6, 37, 41], 4, 5]:
        env.step(action)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [[6, 50, 33], 1, 1, 1, 1, 1, 1, 1, [6, 44, 42], 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, [6, 24, 18], 2, 2, 2, 2, 2, 2, 3, [6, 45, 18], 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, [6, 12, 30], 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 5]:
        env.step(action)


def play_level_5(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [[6, 33, 21], 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, [6, 48, 33], 1, 1, 3, 3, 3, 3, 3, 3, [6, 24, 33], 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, [6, 33, 43], 1, 1, 3, 3, 3, 3, 3, 3, 5]:
        env.step(action)


def play_level_6(env):
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=7.5)
    align_vertical(env, moving_color=9, reference_color=6, center_offset=28.5)
    select_object(env, color=15)
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-7.5)
    align_vertical(env, moving_color=9, reference_color=6, center_offset=22.5)
    select_object(env, color=8)
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-15)
    align_vertical(env, moving_color=9, reference_color=6, center_offset=25.5)
    select_object(env, color=8)
    align_vertical(env, moving_color=9, reference_color=6, center_offset=6)
    activate(env)
