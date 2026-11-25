# Per-level players only compose reusable legs.
from legs import (ascend_breakable_shaft, ascend_with_ceiling_hazards,
                  navigate_consumable_gravity_switches, navigate_scrolling_gravity_maze,
                  navigate_growing_supports, descend_switch_shaft)


def play_level_1(env):
    ascend_breakable_shaft(env)


def play_level_2(env):
    ascend_with_ceiling_hazards(env)


def play_level_3(env):
    ascend_with_ceiling_hazards(env, toggle_color=12)


def play_level_4(env):
    navigate_consumable_gravity_switches(env)


def play_level_5(env):
    navigate_consumable_gravity_switches(
        env, origin=(3, 3), shape=(10, 10), turn_back_after=2)
    ascend_with_ceiling_hazards(
        env, origin=(3, 3), shape=(10, 10), toggle_color=12)


def play_level_6(env):
    navigate_consumable_gravity_switches(env, turn_back_after=2)
    ascend_with_ceiling_hazards(env, toggle_color=12)
    navigate_consumable_gravity_switches(env, toggle_color=12)


def play_level_7(env):
    navigate_scrolling_gravity_maze(env)


def play_level_8(env):
    navigate_growing_supports(env)


def play_level_9(env):
    navigate_growing_supports(env, breakable_color=14)
    descend_switch_shaft(env)
