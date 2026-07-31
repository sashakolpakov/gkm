import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CTL = (6, 46, 22)
B_CTL = (6, 56, 22)
C = (6, 50, 28)
D = (6, 56, 28)
E = (6, 52, 42)
F = (6, 52, 46)
PLATFORM_PATHS = (
    (),
    (C,), (D,),
    (C, C), (D, D),
    (C, C, C), (D, D, D),
    (C, C, C, D),
    (C, C, C, D, D),
    (C, C, C, D, D, D),
)
POCKET = {
    (30, 26), (30, 28),
    (32, 26), (32, 28),
    (34, 26), (34, 28), (34, 30), (34, 32),
    (36, 26), (36, 28), (36, 30), (36, 32),
}


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    tested = 0
    for a_phase in range(6):
        for b_phase in range(8):
            for platform_index, platform_path in enumerate(PLATFORM_PATHS):
                for e_phase in range(2):
                    for f_phase in range(2):
                        node = env.clone()
                        config = (
                            [A_CTL] * a_phase + [B_CTL] * b_phase +
                            list(platform_path) + [E] * e_phase + [F] * f_phase)
                        for action in config:
                            step(node, action)
                        tested += 1
                        if int(node.levels_completed) > base_level:
                            print("REWARD", a_phase, b_phase, platform_index,
                                  e_phase, f_phase, config)
                            return
                        # Put E in its upper position, enter every edge of the
                        # pocket, and attempt to cross each wall.
                        route = [3]
                        if e_phase == 0:
                            route.append(E)
                        route += [
                            3, 3, 3,
                            1, 1, 1,
                            4, 1, 3,
                            2, 2, 2, 2,
                            4, 4, 4, 4,
                        ]
                        for index, action in enumerate(route):
                            step(node, action)
                            pos = avatar(node)
                            if pos not in POCKET:
                                print("ESCAPE", a_phase, b_phase,
                                      platform_index, e_phase, f_phase,
                                      "AT", index, action, pos,
                                      "CONFIG", config, "ROUTE", route[:index + 1])
                                return
                        if tested % 240 == 0:
                            print("PROGRESS", tested)
    print("NONE", tested)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
