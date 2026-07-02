# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1: one target that needs a single rotation change before the avatar
    # can stand on it. The generic search leg discovers the whole plan itself.
    advance_one_level(env)


def play_level_3(env):
    # Level 3 is (like level 1) driven by the avatar's own configuration/position:
    # the generic BFS over avatar states finds the clearing path, and the compact
    # avatar-only state_key is both sufficient (a found path is simulator-verified)
    # and far cheaper than full_state_key here.
    advance_one_level(env)


def play_level_4(env):
    # Level 4 is another avatar-configuration level (like 1 and 3): despite ~110
    # sprites, all but a handful are static maze/wall cells, so avatar position +
    # (shape, colour, rotation) + done-mask is the whole state that matters. The
    # generic BFS over avatar states finds a 43-move clearing path with the cheap
    # avatar-only state_key; any path it returns is simulator-verified.
    advance_one_level(env)


def play_level_2(env):
    # Level 2 is an object-moving level: the goal depends on where sprites end
    # up (the avatar carries them through the maze), so avatar position alone is
    # not the whole state. Search over the full sprite configuration with the
    # same generic BFS leg.
    advance_one_level(env, key_fn=full_state_key)
