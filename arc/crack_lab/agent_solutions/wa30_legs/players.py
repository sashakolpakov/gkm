# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    """Warehouse level: carry each colour-4 box into a target-container cell.
    Fill every cell -> reward fires."""
    fill_targets(env, box_color=4)


def play_level_2(env):
    """Same warehouse carry, now with an autonomous HELPER and a tight step
    budget: the helper carries boxes to the container but can't finish 5 alone in
    time. The avatar delivers 2 boxes into the bin's bottom slots, then parks and
    lets the helper complete the rest -> all boxes on target -> reward fires."""
    fill_bin_with_helper(env, box_color=4, quota=2)


def play_level_3(env):
    """Relay warehouse: a floor-to-ceiling wall (colour 2) splits the map. The
    avatar and its boxes are on one side; an autonomous HELPER (colour 12) ferries
    boxes from the wall into the container on the far side but can't reach the
    avatar's deeper boxes. The avatar relays each box flush against the wall for the
    helper to collect, then idles -> all boxes binned -> reward fires."""
    relay_to_helper(env, box_color=4, wall_color=2)
