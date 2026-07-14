# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # r11l level 1: align the rope's midpoint box onto the ring target.
    # Thin composition: one leg does the whole skill; the mechanics
    # (detect -> move active -> select other -> move) live in legs.py.
    place_box_on_ring(env)
