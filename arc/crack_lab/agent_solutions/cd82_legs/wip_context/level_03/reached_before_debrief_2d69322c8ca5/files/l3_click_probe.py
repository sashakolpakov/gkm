import numpy as np

import gkm_try as harness
from perception import frame_delta


POINTS = {
    "target_F": (3, 3),
    "target_C": (6, 3),
    "target_E": (12, 4),
    "target_8": (7, 9),
    "palette_F": (28, 4),
    "palette_C": (34, 4),
    "canvas_tl": (27, 34),
    "canvas_mid": (32, 39),
    "vessel_top": (31, 20),
    "vessel_body": (31, 25),
    "vessel_edge": (26, 25),
    "background": (20, 20),
}


def compact_delta(before, after):
    delta = frame_delta(before[:-1], after[:-1])
    return delta["count"], delta["bbox"]


def canvas_text(frame):
    chars = "0123456789ABCDEF"
    return "/".join(
        "".join(chars[int(v)] for v in row)
        for row in frame[34:44, 27:37]
    )


def observe(env):
    harness.m.solve(env)
    before = np.asarray(env.frame()).copy()
    for name, point in POINTS.items():
        for prefix in ((), (3,)):
            node = env.clone()
            for action in prefix:
                node.step(action)
            pre = np.asarray(node.frame()).copy()
            node.step(6, *point)
            clicked = np.asarray(node.frame()).copy()
            node.step(5)
            used = np.asarray(node.frame()).copy()
            print(
                name, "at", "N" if not prefix else "NW",
                "click", compact_delta(pre, clicked),
                "use", compact_delta(clicked, used),
                "canvas", canvas_text(used),
                "level", node.levels_completed,
            )
    print("base_unchanged", np.array_equal(before, np.asarray(env.frame())))


if __name__ == "__main__":
    harness.A.run_program("cd82", observe)
