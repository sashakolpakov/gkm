# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def move_vessel_below_and_apply(env):
    """Roll the active vessel below the work tile, then apply its contents."""
    for action in (3, 2, 2, 4, 5):
        env.step(action)
