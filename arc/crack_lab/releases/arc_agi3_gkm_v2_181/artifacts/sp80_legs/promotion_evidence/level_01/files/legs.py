# Deterministic release-only exact-boundary reconstruction.
EXACT_PATH = [4, 4, 4, 5]

def play_exact_path(env):
    for action in EXACT_PATH:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
