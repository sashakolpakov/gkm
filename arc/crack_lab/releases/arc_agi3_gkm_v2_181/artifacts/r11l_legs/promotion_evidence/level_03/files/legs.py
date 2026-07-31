# Deterministic release-only exact-boundary reconstruction.
EXACT_PATH = [[6, 30, 21], [6, 27, 59], [6, 48, 21], [6, 17, 6], [6, 48, 57], [6, 49, 9], [6, 44, 57], [6, 8, 21], [6, 27, 39], [6, 27, 39], [6, 30, 41], [6, 54, 48], [6, 63, 10], [6, 14, 16], [6, 61, 47], [6, 34, 9], [6, 61, 61], [6, 23, 21], [6, 61, 57], [6, 39, 16], [6, 47, 47], [6, 61, 61], [6, 51, 61], [6, 37, 34], [6, 26, 63], [6, 52, 40], [6, 32, 53]]

def play_exact_path(env):
    for action in EXACT_PATH:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
