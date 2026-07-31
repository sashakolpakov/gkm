"""Independent replay check for the discovered level-4 leg path."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from players import play_level_1, play_level_2, play_level_3


PATH = (
    4, 4,
    1, 1, 1, 1, 1,
    3, 1, 3, 2, 3, 1, 3, 2,
    4, 4, 2, 2, 3, 1, 1, 3, 2,
    1, 4, 4, 4, 4, 4, 3, 2, 2, 3, 1, 1, 3, 2,
    4, 2, 4, 1,
)


def inspect(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)
    before = env.levels_completed
    for index, action in enumerate(PATH, 1):
        env.step(action)
        if env.levels_completed > before:
            print("REWARD", index, action, env.levels_completed)
            break
    print("VERIFY", before, env.levels_completed, len(PATH))


if __name__ == "__main__":
    levels, path, err = A.run_program("sk48", inspect)
    print("END", levels, len(path), err)
