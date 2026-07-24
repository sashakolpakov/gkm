# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

UP, DOWN, LEFT, RIGHT, SELECT = 1, 2, 3, 4, 6


def move_steps(env, direction, count):
    """Move the currently selected object repeatedly in one direction."""
    for _ in range(count):
        env.step(direction)


def select_at(env, x, y):
    """Select the object occupying screen coordinate (x, y)."""
    env.step(SELECT, x, y)


def probe_level(env):
    """Temporary compact observation probe for a newly reached level."""
    import numpy as np
    frame = np.asarray(env.frame())
    chars = {0: "o", 1: ".", 2: "B", 4: "T", 5: "x", 14: "O", 15: "#"}
    for r in range(1, 63, 3):
        line = []
        for c in range(1, 63, 3):
            vals, counts = np.unique(frame[r-1:r+2, c-1:c+2], return_counts=True)
            line.append(chars[int(vals[int(np.argmax(counts))])])
        print("MAP", "".join(line))
