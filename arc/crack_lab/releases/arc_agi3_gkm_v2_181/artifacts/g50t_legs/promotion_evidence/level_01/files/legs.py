# Deterministic release-only exact-boundary reconstruction.
EXACT_PATH = [4, 4, 4, 4, 5, 2, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4]

def play_exact_path(env):
    for action in EXACT_PATH:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
