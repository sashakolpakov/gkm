# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import clear_binary_program


def play_level_1(env):
    clear_binary_program(
        env,
        controls=((21, 42), (26, 42), (31, 42), (36, 42), (41, 42),
                  (21, 45), (26, 45), (31, 45), (36, 45), (41, 45)),
        trigger=(36, 55),
    )
