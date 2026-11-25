# Per-level players only compose reusable legs.
from legs import reunite_mirrored_pair


def play_level_1(env):
    reunite_mirrored_pair(env)


def play_level_2(env):
    reunite_mirrored_pair(env, hazard_colors=(8,))
