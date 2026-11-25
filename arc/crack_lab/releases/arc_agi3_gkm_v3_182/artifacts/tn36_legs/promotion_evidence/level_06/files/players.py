# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import clear_binary_program, write_binary_program


def play_level_1(env):
    clear_binary_program(
        env,
        controls=((21, 42), (26, 42), (31, 42), (36, 42), (41, 42),
                  (21, 45), (26, 45), (31, 45), (36, 45), (41, 45)),
        trigger=(36, 55),
    )


def play_level_2(env):
    clear_binary_program(
        env,
        controls=((39, 33), (39, 48), (44, 33), (44, 48),
                  (49, 33), (49, 48), (54, 33), (54, 48)),
        trigger=(46, 58),
    )



def play_level_3(env):
    clear_binary_program(
        env,
        controls=((34, 33), (34, 48), (39, 36), (44, 36),
                  (49, 36), (54, 36), (59, 33), (59, 48)),
        trigger=(57, 58),
    )


def play_level_4(env):
    clear_binary_program(
        env,
        controls=((34, 33), (34, 42), (39, 36), (39, 48),
                  (44, 33), (44, 36), (49, 33), (49, 36),
                  (54, 33), (54, 36), (59, 33), (59, 36)),
        trigger=(57, 58),
    )


def play_level_5(env):
    clear_binary_program(
        env,
        controls=((34, 33), (34, 36), (39, 33), (39, 36),
                  (44, 33), (44, 36), (49, 42),
                  (54, 33), (54, 36), (54, 39), (54, 42),
                  (54, 45), (54, 48), (59, 33), (59, 39)),
        trigger=(57, 58),
    )


def play_level_6(env):
    clear_binary_program(
        env,
        controls=((34, 36), (39, 36), (44, 36), (49, 36), (54, 36),
                  (59, 36), (59, 39), (59, 42), (59, 45), (59, 48)),
        trigger=(57, 58),
    )
    write_binary_program(
        env,
        columns=(34, 39, 44, 49, 54, 59),
        rows=(33, 36, 39, 42, 45, 48),
        words=(62, 30, 30, 1, 1, 1),
        trigger=(57, 58),
    )
    write_binary_program(
        env,
        columns=(34, 39, 44, 49, 54, 59),
        rows=(33, 36, 39, 42, 45, 48),
        words=(30, 30, 30, 30, 61, 1),
        trigger=(57, 58),
    )
    write_binary_program(
        env,
        columns=(34, 39, 44, 49, 54, 59),
        rows=(33, 36, 39, 42, 45, 48),
        words=(62, 62, 62, 60, 62, 1),
        trigger=(57, 58),
    )
