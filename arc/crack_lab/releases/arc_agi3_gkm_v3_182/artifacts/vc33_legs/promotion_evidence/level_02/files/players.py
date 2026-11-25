# Each player composes reusable legs only.
from legs import align_horizontal_indicator, align_indicator_with_transfers


def play_level_1(env):
    align_horizontal_indicator(env)


def play_level_2(env):
    align_indicator_with_transfers(env)
