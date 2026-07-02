# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Deflect the poured stream off both bar edges into the cup openings.
    left = best_deflect_left_col(env.frame())
    move_bar_to_left_col(env, left)
    pour(env)
