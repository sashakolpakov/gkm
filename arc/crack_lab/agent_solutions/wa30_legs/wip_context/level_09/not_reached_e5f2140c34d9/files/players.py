# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    fill_three_slot_target(env)


def play_level_2(env):
    fill_bottom_slots_alongside_courier(env)


def play_level_3(env):
    feed_three_blocks_to_courier(env)


def play_level_4(env):
    feed_two_blocks_to_each_courier(env)


def play_level_5(env):
    expedite_single_courier_with_four_blocks(env)


def play_level_6(env):
    recover_two_blocks_across_courier_port(env)


def play_level_7(env):
    stage_and_recover_two_courier_deliveries(env)


def play_level_8(env):
    disable_competing_couriers_and_expedite_paired_depots(env)

