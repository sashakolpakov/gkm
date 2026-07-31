import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


def counts(node):
    grid = arr(node.frame())
    return int((grid == 11).sum()), int((grid[33:, :] == 11).sum())


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    root = env.clone()
    variants = [("base", SEED)]
    for index in range(len(SEED)):
        variants.append((f"delete_{index}", SEED[:index] + SEED[index + 1 :]))
        for action in (1, 2, 3, 4):
            if action != SEED[index]:
                variants.append((
                    f"replace_{index}_{action}",
                    SEED[:index] + [action] + SEED[index + 1 :],
                ))
            variants.append((
                f"insert_{index}_{action}",
                SEED[:index] + [action] + SEED[index:],
            ))

    painted = {}
    best = (4, None)
    for name, path in variants:
        node = root.clone()
        step_many(node, path)
        total, lower = counts(node)
        if total < 50:
            continue
        shape = selected_shape(node)
        if shape is not None:
            signature = (shape[2], shape[3])
            painted.setdefault(signature, (name, path, shape))
        for down in range(1, 16):
            node.step(2)
            total, lower = counts(node)
            if lower > best[0]:
                best = (lower, name, down, total, selected_shape(node), path)
            if total >= 50 and lower > 4:
                print("FOUND", name, down, total, lower, path)
                print("SHAPE", selected_shape(node))
                return
    print("PAINTED_GEOMETRIES", painted)
    print("BEST", best)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
