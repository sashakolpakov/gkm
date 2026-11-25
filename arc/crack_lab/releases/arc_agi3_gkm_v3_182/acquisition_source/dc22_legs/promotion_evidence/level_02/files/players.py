# Per-level players only compose reusable legs.
from legs import navigate_switchable_platforms


def play_level_1(env):
    navigate_switchable_platforms(env, switches=((48, 20), (48, 36)))


def play_level_2(env):
    navigate_switchable_platforms(env, switches=((52, 22), (52, 40)),
                                  waypoint=(54, 18), max_depth=32)
    navigate_switchable_platforms(env,
                                  switches=((52, 22), (52, 30), (52, 40)))
