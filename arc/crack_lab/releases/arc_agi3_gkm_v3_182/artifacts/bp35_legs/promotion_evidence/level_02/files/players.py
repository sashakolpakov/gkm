# Per-level players only compose reusable legs.
from legs import ascend_breakable_shaft, ascend_with_ceiling_hazards


def play_level_1(env):
    ascend_breakable_shaft(env)


def play_level_2(env):
    ascend_with_ceiling_hazards(env)
