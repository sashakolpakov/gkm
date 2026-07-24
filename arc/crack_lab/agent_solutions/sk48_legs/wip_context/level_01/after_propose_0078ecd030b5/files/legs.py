# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, RETRACT, EXTEND = 1, 2, 3, 4


def move_vertical_lanes(env, direction, lanes):
    """Move the avatar and any attached horizontal train by whole lanes."""
    for _ in range(lanes):
        if env.terminal():
            return
        env.step(direction)


def extend_tether(env, steps):
    """Extend the tether, pushing its attached train toward a new token."""
    for _ in range(steps):
        if env.terminal():
            return
        env.step(EXTEND)


def retract_tether(env, steps):
    """Retract the tether, pulling a contacted token into the train."""
    for _ in range(steps):
        if env.terminal():
            return
        env.step(RETRACT)
