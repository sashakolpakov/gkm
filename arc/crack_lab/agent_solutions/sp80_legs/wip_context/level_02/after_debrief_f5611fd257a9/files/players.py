# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    """Level 1: move the 9-block right 3 times then USE."""
    bfs_or_fallback(env, make_bbox_key(9), fallback=[4, 4, 4, 5],
                    actions=(1, 2, 3, 4, 5, 6), max_states=500, max_depth=20)


def play_level_2(env):
    """Level 2: three movable blocks selected by clicking (action 6 = grab
    the object under the cursor), then moved with the directional actions.

    Blocks at level start:
        A (12 wide) top-left  ~ centred at (col 14, row 18)
        B (12 wide) mid       ~ centred at (col 34, row 26)
        C (20 wide) lower      ~ centred at (col 30, row 38)
    Blocks slide freely (they pass through one another) and are bounded to
    rows 16..51.  The level completes after a final USE(5) once a 12-wide
    block sits at cols 20-31 and the 20-wide block C sits at cols 28-47.
    We drive that configuration: A -> cols20-31, B -> cols36-47,
    C -> cols28-47, then USE.
    """
    grab_and_move(env, 14, 18, [4, 4, 4])  # grab A -> cols 20-31
    grab_and_move(env, 34, 26, [4, 4])     # grab B -> cols 36-47
    grab_and_move(env, 30, 38, [4, 4])     # grab C -> cols 28-47
    play_fixed_sequence(env, [5])          # USE -> commit / win
