# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def prime_board(env, action=1):
    """Spend the board's initial inert input so subsequent controls take effect."""
    env.step(action)


def select_grid_cells_of_color(env, xs, ys, color, click_action=6):
    """Click the sampled cells of a coordinate grid that currently match color."""
    frame = env.frame()
    points = [
        (x, y)
        for y in ys
        for x in xs
        if int(frame[y][x]) == color
    ]
    for x, y in points:
        env.step(click_action, x, y)


def move_until_level_progress(env, action, max_steps):
    """Repeat a movement while bounded, stopping as soon as the level advances."""
    starting_level = env.levels_completed
    for _ in range(max_steps):
        if env.terminal() or env.levels_completed > starting_level:
            break
        env.step(action)
