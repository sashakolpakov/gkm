# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, LEFT, RIGHT, SELECT = 1, 2, 3, 4, 6


def move_steps(env, direction, count):
    """Move the currently selected object repeatedly in one direction."""
    for _ in range(count):
        env.step(direction)


def select_at(env, x, y):
    """Select the object occupying screen coordinate (x, y)."""
    env.step(SELECT, x, y)
