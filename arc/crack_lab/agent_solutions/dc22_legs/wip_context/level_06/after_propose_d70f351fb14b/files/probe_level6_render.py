import importlib.util
import os
import sys

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
D_CONTROLS = {
    "u": (1, (6, 50, 32), 2),
    "r": (4, (6, 54, 36), 3),
    "l": (3, (6, 46, 36), 4),
}

REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11

HUB = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL]
)

TOP_PATH = ["u", "r", "u", "u", "l", "l", "u", "u", "u"]


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def move_ring(env, labels):
    for label in labels:
        outward, control, inward = D_CONTROLS[label]
        apply(env, [outward, control, inward])


def run(env):
    solver.solve(env)
    initial = env.frame()
    mixed = []
    for row in range(31):
        for col in range(20):
            block = initial[
                2 * row:2 * row + 2, 2 * col:2 * col + 2
            ]
            values = tuple(int(value) for value in block.ravel())
            if len(set(values)) > 1:
                mixed.append((row, col, values))
    print("MIXED", mixed, flush=True)
    frames = [env.frame().copy()]
    apply(env, HUB)
    frames.append(env.frame().copy())
    move_ring(env, TOP_PATH)
    frames.append(env.frame().copy())

    palette = [
        "#000000", "#0074D9", "#FF4136", "#2ECC40",
        "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B",
        "#7FDBFF", "#870C25", "#B10DC9", "#39CCCC",
        "#01FF70", "#85144B", "#FFFFFF", "#777777",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, frame, title in zip(
        axes, frames, ("initial", "hub", "ring at north"),
    ):
        axis.imshow(frame, cmap=ListedColormap(palette), vmin=0, vmax=15)
        axis.set_title(title)
        axis.set_xticks(np.arange(-0.5, 64, 2), minor=True)
        axis.set_yticks(np.arange(-0.5, 64, 2), minor=True)
        axis.grid(which="minor", color="#333333", linewidth=0.2)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig("level6_states.png", dpi=180)
    plt.close(figure)


A.run_program("dc22", run)
