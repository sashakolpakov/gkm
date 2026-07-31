# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Ascend the block tower, then step onto the prize revealed at its top.
    climb_to_prize(env)


def play_level_2(env):
    # Taller tower with fatal ceiling hazards, so the greedy ascent walks into
    # dead-end shafts: plan the climb by search instead, then commit it.
    climb_by_search(env)


def play_level_3(env):
    # Nearby hazards must be made safe before each timed crossing.
    climb_local_hazards(env)
