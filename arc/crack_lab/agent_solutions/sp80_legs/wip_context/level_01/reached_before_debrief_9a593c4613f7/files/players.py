# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    """Level 1: move the 9-block right 3 times then USE."""
    # BFS discovers the win in < 50 states; fall back to known sequence.
    import numpy as np

    def _key(e):
        f = np.array(e.frame())
        nines = np.where(f == 9)
        if len(nines[0]) == 0:
            return None
        return (int(nines[0].min()), int(nines[0].max()),
                int(nines[1].min()), int(nines[1].max()))

    path = bfs_win_compact(env, _key, actions=(1, 2, 3, 4, 5, 6),
                           max_states=500, max_depth=20)
    if path:
        play_fixed_sequence(env, path)
    else:
        # Hardcoded fallback: RIGHT RIGHT RIGHT USE
        play_fixed_sequence(env, [4, 4, 4, 5])
