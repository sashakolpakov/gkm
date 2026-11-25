# Per-level players only compose reusable legs.
from legs import (ascend_breakable_shaft, ascend_with_ceiling_hazards,
                  navigate_consumable_gravity_switches)


def play_level_1(env):
    ascend_breakable_shaft(env)


def play_level_2(env):
    ascend_with_ceiling_hazards(env)


def play_level_3(env):
    ascend_with_ceiling_hazards(env, toggle_color=12)


def play_level_4(env):
    navigate_consumable_gravity_switches(env)
