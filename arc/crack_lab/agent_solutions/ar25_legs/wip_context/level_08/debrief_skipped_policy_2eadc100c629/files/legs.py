# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def repeat_action(env, action, count):
    """Take the same discrete action a fixed number of times."""
    for _ in range(count):
        if env.terminal():
            return
        env.step(action)
