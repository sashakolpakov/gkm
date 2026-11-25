# Per-level players only compose reusable legs.
from legs import center_link_on_ring, center_link_groups_on_rings


def play_level_1(env):
    center_link_on_ring(env)


def play_level_2(env):
    center_link_groups_on_rings(env)


def play_level_3(env):
    center_link_groups_on_rings(env)
