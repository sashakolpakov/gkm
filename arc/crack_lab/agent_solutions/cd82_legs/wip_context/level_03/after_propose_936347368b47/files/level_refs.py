import numpy as np

import gkm_try as harness
import players


def text(values):
    chars = "0123456789ABCDEF"
    return "/".join("".join(chars[int(v)] for v in row) for row in values)


def observe(env):
    print("stage1", env.levels_completed, text(np.asarray(env.frame())[3:13, 3:13]))
    players.play_level_1(env)
    frame = np.asarray(env.frame())
    print("stage2", env.levels_completed, text(frame[3:13, 3:13]))
    print("blank", text(frame[34:44, 27:37]))
    for action in (5,):
        env.step(action)
    env.step(6, 46, 4)
    for action in (4, 2, 2):
        env.step(action)
    frame = np.asarray(env.frame())
    print("before_goal", env.levels_completed, text(frame[34:44, 27:37]))


harness.A.run_program("cd82", observe)
