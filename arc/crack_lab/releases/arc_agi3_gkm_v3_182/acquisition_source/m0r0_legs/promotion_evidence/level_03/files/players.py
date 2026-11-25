# Per-level players only compose reusable legs.
from legs import reunite_mirrored_pair, park_selectable_blockers


def play_level_1(env):
    reunite_mirrored_pair(env)


def play_level_2(env):
    reunite_mirrored_pair(env, hazard_colors=(8,))


def play_level_3(env):
    park_selectable_blockers(env)
    reunite_mirrored_pair(env, hazard_colors=(8,))
