"""Dense shape-progress probes for the HUD, black object, and patrol."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_level5 import reach_level_5


BLACK_FROM_RIGHT = (1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3)
BLACK_FROM_BELOW = (1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1)
BLACK_FROM_ABOVE = (1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 3, 2)
COLLISION = (1, 3, 3, 3, 2, 3, 3, 1, 3)
BLACK_THEN_COLLISION = BLACK_FROM_BELOW + (2, 2, 2, 2, 2, 4, 4)
BLACK_REFILL_COLLISION = (
    BLACK_FROM_BELOW
    + (2, 2, 2, 2, 2, 2, 4, 2)
    + (1, 1, 4)
)
LOWER_TO_BLACK = (
    1, 4, 4, 4, 4, 1, 4, 4, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1,
)
ROTATED_TO_BLACK = {
    "shape_1_black": COLLISION + (2, 2, 3) + LOWER_TO_BLACK,
    "shape_2_black": COLLISION + (4, 2, 2, 3, 3) + LOWER_TO_BLACK,
    "shape_3_black": COLLISION + (4, 3, 2, 2, 3) + LOWER_TO_BLACK,
    "shape_0_black": COLLISION + (4, 3, 3, 2, 2) + LOWER_TO_BLACK,
}
TARGET_HUD = {
    (1, 1), (1, 3), (2, 1), (2, 2), (3, 2), (3, 3),
}


def hud_shape(frame):
    crop = np.asarray(frame)[53:63, 1:11]
    blocks = crop.reshape(5, 2, 5, 2).transpose(0, 2, 1, 3)
    cells = []
    for row in range(5):
        for col in range(5):
            values, counts = np.unique(blocks[row, col], return_counts=True)
            color = int(values[int(np.argmax(counts))])
            if color in {8, 9, 12, 14} and int(counts.max()) == 4:
                cells.append((row, col, color))
    return tuple(cells)


def target_shape(frame):
    crop = np.asarray(frame)[4:11, 53:60]
    points = np.argwhere(crop == 8)
    return tuple((int(row), int(col)) for row, col in points)


def run(root, name, path, suffix=()):
    clone = root.clone()
    prior = hud_shape(clone.frame())
    print(name, "start", prior)
    for index, action in enumerate(path + suffix, 1):
        clone.step(action)
        current = hud_shape(clone.frame())
        if current != prior or index == len(path):
            marker = (
                "TARGET_SHAPE"
                if {(row, col) for row, col, _ in current} == TARGET_HUD
                else ""
            )
            print(name, index, action, current, marker)
        prior = current


def inspect(env):
    reach_level_5(env)
    root = env.clone()
    print("target", target_shape(root.frame()))
    run(root, "black_right", BLACK_FROM_RIGHT)
    run(root, "black_below", BLACK_FROM_BELOW)
    run(root, "black_above", BLACK_FROM_ABOVE)
    run(root, "black_right_repeat", BLACK_FROM_RIGHT, (4, 3, 4, 3, 4, 3))
    run(root, "black_below_repeat", BLACK_FROM_BELOW, (2, 1, 2, 1, 2, 1))
    run(root, "black_above_repeat", BLACK_FROM_ABOVE, (1, 2, 1, 2, 1, 2))
    run(
        root,
        "black_refilled_orbit",
        BLACK_FROM_RIGHT,
        (3, 3, 4, 4) + (4, 3) * 8,
    )
    run(root, "collision", COLLISION, (4, 3, 3, 4, 4, 3, 3))
    run(
        root,
        "black_collision",
        BLACK_THEN_COLLISION,
        (4, 3, 3, 4, 4, 3, 3),
    )
    run(
        root,
        "refilled_collision",
        BLACK_REFILL_COLLISION,
        (3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4),
    )
    for name, path in ROTATED_TO_BLACK.items():
        run(root, name, path)
    for contacts in range(2, 8):
        prepared = (
            BLACK_FROM_RIGHT
            + (3, 3, 4, 4)
            + (4, 3) * (contacts - 2)
            + (2, 2, 2, 2, 2, 2, 4, 2)
            + (1, 1, 4)
        )
        run(
            root,
            f"compose_b{contacts}_p",
            prepared,
            (3, 4, 4, 3, 3, 4, 4, 3, 3, 4),
        )


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
