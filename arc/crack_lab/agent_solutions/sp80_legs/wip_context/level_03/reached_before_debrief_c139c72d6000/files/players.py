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


def play_level_3(env):
    """Level 3: same select-then-drive family as level 2, now with FOUR movable
    blocks over three ∏-sockets. Column alignment only (rows irrelevant); a
    final USE(5) commits the configuration.

    Blocks at level start (color 8, one selected turns 9):
        A (16 wide) rows20-23 cols 8-23  -> grab @ (col15,row21)
        B (20 wide) rows28-31 cols40-59  -> grab @ (col49,row29)
        C (24 wide) rows32-35 cols 8-31  -> grab @ (col19,row33)
        N (24 wide) rows40-43 cols36-59  -> grab @ (col47,row41)
    Blocks slide 4px/step, pass through one another, bounded left to col 0.

    Winning config (found by bounded clone random-search over left-columns,
    win density ~0.6% as in level 2): left-cols A=12, B=0, C=28, N=40, USE.
    From the start positions that is: A right x1, B left x10, C right x5,
    N right x1, then USE."""
    grab_and_move(env, 15, 21, [4])            # A -> left col 12
    grab_and_move(env, 49, 29, [3] * 10)       # B -> left col 0
    grab_and_move(env, 19, 33, [4] * 5)        # C -> left col 28
    grab_and_move(env, 47, 41, [4])            # N -> left col 40
    play_fixed_sequence(env, [5])              # USE -> commit / win
