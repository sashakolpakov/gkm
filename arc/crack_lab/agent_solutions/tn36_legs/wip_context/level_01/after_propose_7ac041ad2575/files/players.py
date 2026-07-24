# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    import time
    from perception import arr, connected_components

    base = arr(env.frame()).copy()

    def cyan(frame):
        return [(b.bbox, b.area) for b in connected_components(frame, colors=[11])]

    print("cyan0", cyan(base))
    left = env.clone()
    right = env.clone()
    for depth in range(1, 6):
        left.step(6, 0, 0)
        print("isolation", depth, "left", cyan(left.frame()),
              "right", cyan(right.frame()), "env", cyan(env.frame()))
    for delay in (0.05, 0.10, 0.20):
        time.sleep(delay)
        print("delay", delay, "env", cyan(env.frame()),
              "fresh", cyan(env.clone().frame()))
