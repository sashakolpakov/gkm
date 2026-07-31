import numpy as np

import gkm_try as harness


PLAN = (
    3, 2, (6, 52, 4), 5,
    1, (6, 28, 4), 5,
    4, 4, 2, (6, 46, 4), 5,
    1, 3, 3, (6, 28, 4), 5,
    4, (6, 34, 4), (6, 31, 20),
)


def observe(env):
    harness.m.solve(env)
    target = np.asarray(env.frame())[3:13, 3:13].copy()
    node = env.clone()
    for index, action in enumerate(PLAN, 1):
        node.step(*action) if isinstance(action, tuple) else node.step(action)
        work = np.asarray(node.frame())[34:44, 27:37]
        print(
            index, action,
            "mismatch", int(np.count_nonzero(work != target)),
            "level", node.levels_completed,
        )
    print("EXACT", np.array_equal(np.asarray(node.frame())[34:44, 27:37], target))


if __name__ == "__main__":
    harness.A.run_program("cd82", observe)
