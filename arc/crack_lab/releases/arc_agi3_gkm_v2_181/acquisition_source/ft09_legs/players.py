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
    # Level 4 is the pattern-key board family again, but the blocks are now
    # THREE-STATE cells that cycle 9->8->12 on each click (level 3's blocks were
    # two-state). The mini-keys still decode a target: each marked neighbour takes
    # its key's centre colour and every other block takes one shared default
    # colour. solve_multistate_key_board discovers the click cycle, decodes the
    # per-block target, resolves the single default on clones, and commits.
    solve_multistate_key_board(env)


def play_level_2(env):
    # Level 2 is the SAME toggle-tile board family, only a wider 3-row grid
    # (~13 tiles). No new skill is needed: the shared solve_toggle_board leg
    # probes for the tiles and subset-searches the completing clicks on clones.
    solve_toggle_board(env)


def play_level_5(env):
    # Level 5 adds coupled controls to the two-state pattern-key family. The
    # shared leg discovers every control's effect, decodes clue constraints, and
    # solves their binary parity system before committing the verified prefix.
    solve_coupled_key_board(env)


def play_level_6(env):
    # Level 6 is the coupled binary key-board family in a new layout where the
    # controls themselves carry direction marks. The generalized shared leg
    # discovers their effects, reads only clue symbols adjacent to controls,
    # solves the parity constraints, and commits its clone-verified click plan.
    solve_coupled_key_board(env)
