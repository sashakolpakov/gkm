# Shared leg library: small, named, reusable skills.


def clear_binary_program(env, controls, trigger, lit_color=1):
    """Clear lit binary instruction segments, then execute the program.

    The number of lit segments is the dense progress measure. Coordinates
    identify independent controls, so already cleared segments need no click.
    """
    for x, y in controls:
        if int(env.frame()[y][x]) == lit_color:
            env.step(6, x, y)
    env.step(6, *trigger)


def write_binary_program(env, columns, rows, words, trigger, lit_color=1):
    """Match binary instruction words, restoring cleared bits when needed.

    Each row is one bit, least significant first. The remaining mismatched
    segments provide dense progress while rewriting an existing program.
    """
    for x, word in zip(columns, words):
        for bit, y in enumerate(rows):
            if (int(env.frame()[y][x]) == lit_color) != bool(word & (1 << bit)):
                env.step(6, x, y)
    env.step(6, *trigger)

