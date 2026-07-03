# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def play_sequence(env, actions):
    """Replay a fixed sequence of actions on the real env."""
    for a in actions:
        env.step(a)
