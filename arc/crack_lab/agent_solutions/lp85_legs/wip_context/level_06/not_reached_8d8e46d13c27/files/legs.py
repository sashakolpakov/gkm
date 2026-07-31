# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
def repeat_click(env, x, y, times):
    """Click one coordinate repeatedly, stopping if the current level ends."""
    starting_level = env.levels_completed
    for _ in range(times):
        if env.terminal() or env.levels_completed != starting_level:
            break
        env.step(6, x, y)
