# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # r11l level 1: align the rope's midpoint box onto the ring target.
    place_box_on_ring(env)
