import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def color_boxes(frame, color):
    return [
        (blob.bbox, blob.size)
        for blob in connected_components(frame, colors=(color,), min_area=1)
    ]


def click(env, point):
    env.step(6, point[0], point[1])


def run(env):
    solver.solve(env)
    base = np.asarray(env.frame()).copy()
    print(
        "ENTRY",
        "level", env.levels_completed + 1,
        "actions", list(env.actions),
        "colors", color_counts(base),
        flush=True,
    )
    print("COLOR14", color_boxes(base, 14), flush=True)

    for action in (1, 2, 3, 4):
        clone = env.clone()
        before = np.asarray(clone.frame()).copy()
        clone.step(action)
        after = np.asarray(clone.frame())
        changed = before != after
        transitions = {}
        for old, new in zip(before[changed], after[changed]):
            pair = (int(old), int(new))
            transitions[pair] = transitions.get(pair, 0) + 1
        print(
            "KEY", action,
            "delta", frame_delta(before, after),
            "pairs", sorted(transitions.items()),
            "color14", color_boxes(after, 14),
            "reward", clone.levels_completed,
            flush=True,
        )

    controls = {
        "A": (56, 8),
        "B": (51, 25),
        "S": (51, 48),
        "DU": (50, 32),
        "DR": (54, 36),
        "DL": (46, 36),
    }
    for name, point in controls.items():
        clone = env.clone()
        phases = []
        for _ in range(4):
            before = np.asarray(clone.frame())[:, :40].copy()
            click(clone, point)
            after = np.asarray(clone.frame())[:, :40]
            phases.append(
                (
                    frame_delta(before, after)["count"],
                    frame_delta(before, after)["bbox"],
                    color_boxes(after[:62, :40], 8),
                    color_boxes(after[:62, :40], 14),
                )
            )
        print("CONTROL", name, point, "phases", phases, flush=True)


arena.run_program("dc22", run)
