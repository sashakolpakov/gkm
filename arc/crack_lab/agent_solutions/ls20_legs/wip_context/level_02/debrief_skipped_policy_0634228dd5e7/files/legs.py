# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4


def follow_cardinal_runs(env, runs):
    """Walk a route expressed as compact ``(action, count)`` segments."""
    for action, count in runs:
        for _ in range(count):
            if env.terminal():
                return
            env.step(action)
