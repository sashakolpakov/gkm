"""Per-level players only compose reusable legs."""
from legs import follow_key_runs


def play_level_1(env):
    follow_key_runs(env, ((3, 3), (1, 4), (4, 3), (1, 3)))
