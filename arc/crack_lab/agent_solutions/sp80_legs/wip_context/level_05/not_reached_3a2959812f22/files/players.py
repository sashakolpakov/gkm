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
    drive_objects(env, [
        ((14, 18), [4, 4, 4]),  # grab A -> cols 20-31
        ((34, 26), [4, 4]),     # grab B -> cols 36-47
        ((30, 38), [4, 4]),     # grab C -> cols 28-47
    ], commit=[5])              # USE -> commit / win


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
    plan = [
        ((15, 21), [4]),        # A -> left col 12
        ((49, 29), [3] * 10),   # B -> left col 0
        ((19, 33), [4] * 5),    # C -> left col 28
        ((47, 41), [4]),        # N -> left col 40
    ]
    drive_objects(env, plan, commit=[5])              # USE -> commit / win


def play_level_4(env):
    """Level 4: same select-then-drive-then-commit family as levels 2 and 3.

    FIVE movable blocks (color 8; the selected one is color 9) sit in the upper
    chamber over FOUR Π-sockets at the bottom. Directional actions move the
    currently-selected block (3 px/step, all four directions; blocks pass freely
    through one another with the selected one drawn on top). Only column position
    matters for the win; a single USE(5) then commits the configuration.

    Blocks at level start (name, grab pixel col=x row=y, width, initial left-col):
        A (w15) grab @ (24,18)  left 17
        B (w15) grab @ (45,18)  left 38
        C (w21) grab @ (18,30)  left  8   (two 9-wide parts split by a 4-marker)
        D (w12) grab @ (49,33)  left 44
        E (w12) grab @ (43,42)  left 38

    Winning left-cols (found by bounded clone random-search over left-columns,
    same idiom as level 3): A=26, B=14, C=32, D=14, E=17, then USE. Expressed as
    directional steps from each block's start column below."""
    plan = [
        ((24, 18), [4, 4, 4]),   # A: left 17 -> 26
        ((45, 18), [3] * 8),     # B: left 38 -> 14
        ((18, 30), [4] * 8),     # C: left  8 -> 32
        ((49, 33), [3] * 10),    # D: left 44 -> 14
        ((43, 42), [3] * 7),     # E: left 38 -> 17
    ]
    from perception import connected_components
    def obs(e):
        return [(b.color, b.bbox, b.area) for b in connected_components(e.frame(), colors=(8, 9, 11, 15), min_area=4)]
    c = env.clone(); drive_objects(c, plan, commit=())
    print("L4 relation start", obs(env)); print("L4 relation solved", obs(c))
    drive_objects(env, plan, commit=[5])               # USE -> commit / win


def play_level_5(env):
    """Temporary observational probe for the new level."""
    from perception import connected_components
    def selected(e):
        bs = connected_components(e.frame(), colors=(9,), min_area=4)
        return [(b.bbox, b.area) for b in bs]
    for label, xy, action, steps in (
            ("A-up", (34, 21), 1, 8), ("B-up", (24, 33), 1, 10),
            ("C-up", (48, 33), 1, 10), ("D-left", (35, 43), 3, 10)):
        c = env.clone(); c.step(6, *xy)
        trace = [selected(c)]
        for _ in range(steps):
            c.step(action); trace.append(selected(c))
        print("L5 reach", label, trace)
