# Per-level players only compose reusable legs.
from legs import traverse_linked_tiles


def play_level_1(env):
    traverse_linked_tiles(env)
