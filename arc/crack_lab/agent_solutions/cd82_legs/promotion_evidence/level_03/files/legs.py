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


def apply_west_north_east_north_layers_then_payload(
        env, west_selector, north_selector, east_selector, payload_selector,
        selector_y, payload_x, payload_y):
    """Paint W/NW/E/NW layers, return north, then apply the top payload."""
    for action in (3, 2):
        env.step(action)
    env.step(6, west_selector, selector_y)
    env.step(5)

    env.step(1)
    env.step(6, north_selector, selector_y)
    env.step(5)

    for action in (4, 4, 2):
        env.step(action)
    env.step(6, east_selector, selector_y)
    env.step(5)

    for action in (1, 3, 3):
        env.step(action)
    env.step(6, north_selector, selector_y)
    env.step(5)

    env.step(4)
    env.step(6, payload_selector, selector_y)
    env.step(6, payload_x, payload_y)
