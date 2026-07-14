# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a toggle-tile board (framed 3x3): click a subset of tiles to
    # reach the hidden target configuration that completes the level.
    solve_toggle_board(env)


def play_level_3(env):
    # Level 3 is a 'pattern-key' toggle board: a lattice of solid blocks with a
    # few decorated 'pattern' blocks whose 3x3 mini-keys dictate which
    # neighbouring blocks to toggle. No subset search is feasible (23 tiles), so
    # the target is DECODED from the pattern blocks by solve_pattern_key_board.
    solve_pattern_key_board(env)


def play_level_4(env):
    # Level 4 is a MULTI-STATE pattern-key board: same lattice of clickable
    # blocks + non-clickable pattern keys as level 3, but each block cycles THREE
    # colours per click (9 -> 8 -> 12 -> 9). The decode maps each pattern key's
    # neighbour to a target STATE from (surround-cell colour, centre colour); the
    # exact mapping is discovered by enumerating consistent hypotheses and
    # verifying on clones. Reuses the generic solve_pattern_state_board leg.
    solve_pattern_state_board(env)


def play_level_2(env):
    # Level 2 is the SAME toggle-tile board family, only a wider 3-row grid
    # (~13 tiles). No new skill is needed: the shared solve_toggle_board leg
    # probes for the tiles and subset-searches the completing clicks on clones.
    solve_toggle_board(env)
