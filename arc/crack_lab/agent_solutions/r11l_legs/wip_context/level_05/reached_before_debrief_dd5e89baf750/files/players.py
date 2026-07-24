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


def play_level_4(env):
    # r11l level 4: same rope/box/ring mechanic, but boxes AND rings are
    # MULTI-COLOURED and decoy rings litter the board.  Match each box to the
    # ring with an identical colour palette (discovering rope->box identity by
    # probing), then walk-then-straddle it in.  Thin composition; all mechanics
    # live in legs.py.
    place_multicolor_boxes(env)


def play_level_3(env):
    # r11l level 3: another multi-rope board (more colour systems / farther
    # boxes).  Same skill as level 2 — the generic solver discovers every
    # colour system, walks each far box in, then straddle-snaps it onto its
    # own-colour ring.  Nothing level-specific to add: thin composition.
    place_boxes_on_rings(env)


def play_level_5(env):
    # Gather compatible single-palette solids into each centroid carrier, then
    # deliver the assembled palettes to the exact matching (non-decoy) rings.
    assemble_pieces_on_rings(env)
