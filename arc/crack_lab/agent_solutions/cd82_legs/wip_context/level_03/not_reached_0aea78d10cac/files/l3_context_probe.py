import numpy as np

import gkm_try as harness


PALETTE = {
    "zero": (22, 4),
    "fifteen": (28, 4),
    "twelve": (34, 4),
    "eleven": (40, 4),
    "fourteen": (46, 4),
    "eight": (52, 4),
    "nine": (58, 4),
}


def step_path(env, path):
    node = env.clone()
    for action in path:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    return node


def grid_text(values):
    chars = "0123456789ABCDEF"
    return "/".join("".join(chars[int(v)] for v in row) for row in values)


def summary(base, path):
    node = step_path(base, path)
    frame = np.asarray(node.frame())
    target = frame[3:13, 3:13]
    canvas = frame[34:44, 27:37]
    exact = int(np.count_nonzero(target == canvas))
    nonblank = int(np.count_nonzero(canvas))
    return (
        path,
        node.levels_completed,
        exact,
        nonblank,
        grid_text(canvas),
    )


def observe(env):
    harness.m.solve(env)
    print("target", grid_text(np.asarray(env.frame())[3:13, 3:13]))
    paths = [
        [],
        [5],
        [3],
        [4],
        [3, 5],
        [4, 5],
        [3, 2, 5],
        [4, 2, 5],
        [3, 2, 2, 4, 5],
        [4, 2, 2, 5],
        [3, 5, (6, 34, 4), 5],
        [5, (6, 34, 4), 3, 5],
        [3, 5, (6, 46, 4), 4, 2, 2, 5],
        [4, 2, 2, 5, (6, 52, 4), 3, 2, 2, 5],
        [(6, 34, 4), 5, (6, 28, 4), 5],
        [(6, 46, 4), 5, (6, 28, 4), 5],
        [(6, 52, 4), 5, (6, 28, 4), 5],
        [(6, 46, 4), 5, (6, 34, 4), 5],
        [(6, 52, 4), 5, (6, 46, 4), 5],
    ]
    for name, point in PALETTE.items():
        select = (6, point[0], point[1])
        paths.extend([
            [select],
            [select, 5],
            [select, 3, 5],
            [select, 4, 5],
            [select, 4, 2, 2, 5],
        ])
    for path in paths:
        print(summary(env, path))


harness.A.run_program("cd82", observe)
