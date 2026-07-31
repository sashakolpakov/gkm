# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def walk(env, direction, steps):
    """Translate the controlled figure repeatedly in one cardinal direction."""
    for _ in range(steps):
        env.step(direction)


def rotate_quarter_turns(env, turns):
    """Rotate the controlled figure by the requested number of quarter turns."""
    for _ in range(turns):
        env.step(5)


def select_figure(env, x, y):
    """Transfer control to the figure occupying a screen coordinate."""
    env.step(6, x, y)
