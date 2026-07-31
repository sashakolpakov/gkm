import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components
import players


def square_state(frame):
    squares = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame)
        if (
            b.bbox[0] >= 10
            and b.size[0] == b.size[1]
            and b.area == b.size[0] ** 2
            and b.color not in (3, 4, 5, 7, 9)
        )
    ]
    return squares, color_counts(frame).get(9, 0)


def inspect(env):
    while env.levels_completed < 5:
        level = env.levels_completed + 1
        getattr(players, f"play_level_{level}")(env)
    start = env.levels_completed
    path = [
        (6, 28, 27), (6, 20, 19), (6, 12, 12), (6, 6, 16),
        (6, 56, 57), (6, 56, 57), (6, 56, 57),
    ]
    print("level", start + 1, "actions", env.actions,
          "state", square_state(env.frame()))
    clone = env.clone()
    for index, action in enumerate(path, 1):
        clone.step(*action)
        print("step", index, "level", clone.levels_completed,
              "state", square_state(clone.frame()))
        if clone.levels_completed > start:
            break


A.run_program("su15", inspect)
