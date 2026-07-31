import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from perception import color_counts, connected_components


def summary(env):
    counts = color_counts(env.frame())
    blobs = [
        (b.color, b.bbox, b.area)
        for b in connected_components(env.frame(), colors=(4, 8, 14), min_area=4)
    ]
    return counts, blobs


def probe(env):
    while env.levels_completed < 4:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    path = (
        [5] * 4 + [1] * 3 + [4] * 6 + [(6, 54, 6)] + [5] * 3
        + [(6, 5, 38)] + [1] * 10 + [4] * 10
        + [(6, 47, 47)] + [5] * 3 + [1] * 10 + [4] * 2
    )
    print("start", env.levels_completed, summary(env))
    for index, action in enumerate(path, 1):
        before = env.levels_completed
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)
        if isinstance(action, tuple) or env.levels_completed != before:
            print(index, action, "level", env.levels_completed, summary(env))
        if index == 53:
            frame = env.frame()
            print("before_win_cells")
            for rr in range(0, 64, 3):
                line = ""
                for cc in range(0, 64, 3):
                    value = int(frame[rr, cc])
                    line += {0: "x", 4: "#", 8: "A", 12: "s", 15: "."}.get(value, "?")
                if line.strip("."):
                    print(rr // 3, line)


arena.run_program("cn04", probe)
