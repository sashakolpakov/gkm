"""Small reusable skills using only the public action surface."""


def move(env, direction, steps=1):
    """Walk a controlled object a fixed number of cardinal grid steps."""
    if direction not in (1, 2, 3, 4):
        raise ValueError('direction must be a cardinal key')
    for _ in range(steps):
        if env.terminal():
            return
        env.step(direction)


def select_object(env, x, y):
    """Transfer control by clicking an observed object's center."""
    if not all(isinstance(v, int) and 0 <= v < 64 for v in (x, y)):
        raise ValueError('selection must be inside the frame')
    env.step(6, x, y)
