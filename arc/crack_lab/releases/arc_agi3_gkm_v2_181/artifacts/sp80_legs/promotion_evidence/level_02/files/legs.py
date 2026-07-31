# Deterministic release-only exact-boundary reconstruction.
EXACT_PATH = [4, 4, 4, 5, [6, 14, 18], 4, 4, 4, [6, 34, 26], 4, 4, [6, 30, 38], 4, 4, 5]

def play_exact_path(env):
    for action in EXACT_PATH:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
