# Per-level players only compose reusable legs.
from legs import align_horizontal, activate


def play_level_1(env):
    align_horizontal(env, moving_color=9, reference_color=6, center_offset=-4)
    activate(env)
