"""Small reusable movement and recorded-companion skills."""
from perception import safe_step


def walk_straight(env, action, count):
    """Take up to count steps, stopping at a level transition or terminal."""
    level = env.levels_completed
    for _ in range(count):
        if env.terminal() or env.levels_completed != level:
            break
        safe_step(env, action)


def release_recorded_companion(env):
    """Restart at entry while a companion replays and holds the recorded route."""
    if not env.terminal():
        safe_step(env, 5)
