# Per-level players only compose reusable legs.
from legs import align_selected_cross, align_selected_outline, select_next_cross


def play_level_1(env):
    align_selected_cross(env)
    select_next_cross(env)
    align_selected_cross(env)


def play_level_2(env):
    align_selected_outline(env)
    select_next_cross(env)
    align_selected_outline(env)
    select_next_cross(env)
    align_selected_cross(env)
