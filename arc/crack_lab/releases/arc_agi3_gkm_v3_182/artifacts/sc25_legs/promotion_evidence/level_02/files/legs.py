"""Reusable skills derived from public frames and verified clone experiments."""
from perception import arr, safe_step


def repeat_key(env, key, count):
    """Apply a bounded run of one key, stopping when the level changes."""
    level = env.levels_completed
    for _ in range(count):
        if env.terminal() or env.levels_completed != level:
            break
        safe_step(env, key)


def enter_displayed_pattern(env, ink=15):
    """Copy the displayed 3x3 glyph into its adjacent clickable grid."""
    frame = arr(env.frame()).copy()
    cells = [(row, col) for row in range(3) for col in range(3)
             if int(frame[51 + 3 * row, 12 + 3 * col]) == ink]
    level = env.levels_completed
    for row, col in cells:
        if env.terminal() or env.levels_completed != level:
            break
        safe_step(env, (6, 25 + 5 * col, 50 + 5 * row))
