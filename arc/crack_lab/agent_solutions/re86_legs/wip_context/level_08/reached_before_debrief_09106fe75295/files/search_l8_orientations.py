import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import selected_shape, step_many


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    for index in range(2):
        base = env.clone()
        if index:
            base.step(5)
        found = {}
        for first_down in range(1, 4):
            for left in range(2, 7):
                for second_down in range(1, 4):
                    for up in range(1, 4):
                        path = (
                            [2] * first_down
                            + [3] * left
                            + [2] * second_down
                            + [1] * up
                        )
                        node = base.clone()
                        step_many(node, path)
                        shape = selected_shape(node)
                        if shape is None:
                            continue
                        r0, c0, r1, c1 = shape[3]
                        size = (r1 - r0 + 1, c1 - c0 + 1)
                        if size in ((7, 19), (10, 16)):
                            key = (size, shape[0][0] % 3, shape[0][1] % 3)
                            found.setdefault(key, (path, shape))
        print("index", index, found)

    base = [2] + [3] * 4 + [2] * 2 + [1]
    variants = [("base", base)]
    for position in range(len(base) + 1):
        for action in (1, 2, 3, 4):
            variants.append((
                f"insert_{position}_{action}",
                base[:position] + [action] + base[position:],
            ))
    for position in range(len(base)):
        variants.append((
            f"delete_{position}",
            base[:position] + base[position + 1:],
        ))
        for action in (1, 2, 3, 4):
            variants.append((
                f"replace_{position}_{action}",
                base[:position] + [action] + base[position + 1:],
            ))
    mover = env.clone()
    mover.step(5)
    results = {}
    for name, path in variants:
        node = mover.clone()
        step_many(node, path)
        shape = selected_shape(node)
        if shape is None:
            continue
        r0, c0, r1, c1 = shape[3]
        if (r1 - r0 + 1, c1 - c0 + 1) == (7, 19):
            key = (shape[0][0] % 3, shape[0][1] % 3)
            results.setdefault(key, (name, path, shape))
    print("mutations", results)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
