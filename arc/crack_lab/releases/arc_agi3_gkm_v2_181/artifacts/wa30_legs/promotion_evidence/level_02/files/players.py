# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    fill_three_slot_target(env)


def play_level_2(env):
    fill_bottom_slots_alongside_courier(env)
