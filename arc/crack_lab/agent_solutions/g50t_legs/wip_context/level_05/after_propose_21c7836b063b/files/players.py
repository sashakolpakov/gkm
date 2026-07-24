# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # g50t level 1: move the avatar to the goal region, using USE to open the
    # gate that unlocks the far side of the maze.  Fully handled by the general
    # unlock-then-reach leg (bounded-depth, grow-pruned toggle search).
    solve_unlock_reach(env)


def play_level_2(env):
    # g50t level 2: same "reach the goal chamber" objective, but the avatar's
    # region is only linked to the chamber through gates that open remotely when
    # the avatar walks over wall segments; a USE resets the avatar to start
    # while keeping opened gates.  A single USE does not enlarge reachability, so
    # the plain unlock leg fails here -- the general USE-macro unlock search
    # chains several staged openings until the chamber is reachable.
    solve_unlock_macro(env)


def play_level_3(env):
    # g50t level 3: another staged-unlock maze of the same family as level 2 --
    # the goal only becomes movement-reachable after chaining several
    # walk-somewhere-then-USE openings.  No new skill is needed: the SAME
    # general USE-macro unlock leg subsumes this level too, so this player is
    # the identical thin composition as play_level_2.
    solve_unlock_macro(env)


def play_level_4(env):
    # g50t level 4: yet another staged-unlock maze of the level-2/3 family --
    # the goal chamber only becomes movement-reachable after chaining several
    # walk-somewhere-then-USE openings.  No new skill: the SAME general USE-macro
    # unlock leg clears it, so this player is the identical thin composition as
    # play_level_2 / play_level_3.
    solve_unlock_macro(env)


def play_level_5(env):
    # g50t level 5 preserves the staged-unlock mechanic: walking onto special
    # wall segments changes the board, and USE returns the avatar to its start
    # while retaining those changes.  Compose the existing general macro leg.
    solve_unlock_macro(env)
