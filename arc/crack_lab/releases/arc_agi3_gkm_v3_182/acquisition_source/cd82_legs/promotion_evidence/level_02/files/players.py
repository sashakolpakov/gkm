# Per-level players compose reusable legs.
from legs import paint_half, finish_reference_layer


def play_level_1(env):
    paint_half(env, 'bottom')


def play_level_2(env):
    paint_half(env, 'top')
    finish_reference_layer(env, 12)
