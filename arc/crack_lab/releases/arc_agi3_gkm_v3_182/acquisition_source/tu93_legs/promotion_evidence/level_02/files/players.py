# Per-level players only compose reusable legs.
from legs import clear_linked_sentries, traverse_linked_tiles


def play_level_1(env):
    traverse_linked_tiles(env)


def play_level_2(env):
    clear_linked_sentries(env)
    traverse_linked_tiles(env)
