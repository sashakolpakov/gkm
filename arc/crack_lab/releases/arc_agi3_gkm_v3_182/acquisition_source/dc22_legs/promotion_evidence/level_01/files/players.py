# Per-level players only compose reusable legs.
from legs import navigate_switchable_platforms


def play_level_1(env):
    navigate_switchable_platforms(env, switches=((48, 20), (48, 36)))
