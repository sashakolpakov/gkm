# Per-level players only compose reusable legs.
from legs import align_selected_cross, select_next_cross


def play_level_1(env):
    align_selected_cross(env)
    select_next_cross(env)
    align_selected_cross(env)
