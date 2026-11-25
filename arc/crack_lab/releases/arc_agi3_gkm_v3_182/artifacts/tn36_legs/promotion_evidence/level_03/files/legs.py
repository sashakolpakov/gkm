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

