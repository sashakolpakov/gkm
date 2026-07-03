# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa

_L1 = [1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,2,3,1,3,2,3,1,4,4,4,1,1,1]

def play_level_1(env):
    play_sequence(env, _L1)
