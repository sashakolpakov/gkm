import itertools

import numpy as np

import gkm_try as harness


MASK_PATHS = {
    "N": [],
    "NW": [3],
    "NE": [4],
    "W": [3, 2],
    "E": [4, 2],
    "SW": [3, 2, 2],
    "SE": [4, 2, 2],
    "S": [3, 2, 2, 4],
}
COLORS = (15, 12, 14, 8)


def moved(env, path):
    node = env.clone()
    for action in path:
        node.step(action)
    return node


def observe(env):
    harness.m.solve(env)
    frame = np.asarray(env.frame())
    target = frame[3:13, 3:13]
    masks = {}
    for name, path in MASK_PATHS.items():
        node = moved(env, path)
        node.step(5)
        masks[name] = np.asarray(node.frame())[34:44, 27:37] != 0
    for color in COLORS:
        region = target == color
        print(
            "coverage", color,
            [
                (name, int(np.count_nonzero(region & mask)), int(region.sum()))
                for name, mask in masks.items()
                if np.all(mask[region])
            ],
        )

    solutions = []
    for order in itertools.permutations(COLORS):
        for names in itertools.product(MASK_PATHS, repeat=4):
            canvas = np.zeros((10, 10), dtype=np.int8)
            for color, name in zip(order, names):
                canvas[masks[name]] = color
            if np.array_equal(canvas, target):
                solutions.append(list(zip(order, names)))
    print("solutions", len(solutions))
    for solution in solutions[:20]:
        print(solution)


harness.A.run_program("cd82", observe)
