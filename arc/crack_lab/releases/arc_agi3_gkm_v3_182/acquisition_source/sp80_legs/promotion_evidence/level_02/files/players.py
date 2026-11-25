# Per-level players only compose reusable legs.
from legs import align_horizontal, activate, select_object


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
