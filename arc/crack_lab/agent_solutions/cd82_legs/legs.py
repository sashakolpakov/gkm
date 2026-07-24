# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def move_vessel_below_and_apply(env):
    """Roll the active vessel below the work tile, then apply its contents."""
    for action in (3, 2, 2, 4, 5):
        env.step(action)


def apply_current_then_select_and_apply_southeast(env, selector_x, selector_y):
    """Apply the current top stamp, then select and apply a southeast stamp."""
    env.step(5)
    env.step(6, selector_x, selector_y)
    for action in (4, 2, 2, 5):
        env.step(action)
