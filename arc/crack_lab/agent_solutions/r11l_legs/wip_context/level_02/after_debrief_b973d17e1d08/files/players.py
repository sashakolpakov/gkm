# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # r11l level 1: align the rope's midpoint box onto the ring target.
    # Thin composition: one leg does the whole skill; the mechanics
    # (detect -> move active -> drag_endpoint the other) live in legs.py.
    place_box_on_ring(env)


def play_level_2(env):
    # r11l level 2: several colour-coded rope systems sharing one cursor.
    # Each colour's box (centroid of its endpoints) must land on its own-colour
    # ring.  A far box is rope-walked; a near one is straddled exactly.
    # All mechanics live in legs.py; this stays a thin composition.
    place_boxes_on_rings(env)
