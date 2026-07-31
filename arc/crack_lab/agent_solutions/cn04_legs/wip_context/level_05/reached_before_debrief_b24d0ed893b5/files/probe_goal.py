import gkm_try as harness
import players
from perception import color_counts, connected_components, object_candidates


def summary(env, label):
    frame = env.frame()
    non_bg = [b for b in connected_components(frame, min_area=4) if b.color != 15]
    print(label, "level", env.levels_completed, "colors", color_counts(frame))
    bg = max(color_counts(frame), key=color_counts(frame).get)
    print("occupied", int((frame[1:] != bg).sum()), "background", bg)
    print("blobs", [(b.color, b.bbox, b.area) for b in non_bg])


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    summary(env, "L4_START")
    path = [
        [6, 36, 21], 5, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        [6, 45, 42], 5, 5, 5, 1, 1, 1, 1,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        [6, 18, 48], 5, 1, 1, 1, 1, 1, 3, 3,
    ]
    for action in path:
        env.step(action)
    summary(env, "BEFORE_FINAL")
    env.step(3)
    summary(env, "AFTER_FINAL")


harness.A.run_program("cn04", probe)
