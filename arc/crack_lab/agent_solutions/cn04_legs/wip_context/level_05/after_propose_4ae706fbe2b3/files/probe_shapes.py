import numpy as np

import gkm_try as harness


CHARS = {0: "O", 4: "#", 8: "x", 10: "B", 11: "A", 14: "C", 15: "."}


def render_region(frame, bbox):
    r0, c0, r1, c1 = bbox
    rows = []
    for r in range(r0 + 1, r1 + 1, 3):
        rows.append("".join(CHARS.get(int(frame[r, c]), "?")
                            for c in range(c0 + 1, c1 + 1, 3)))
    return "/".join(rows)


def probe(env):
    harness.resumed_solve(env)
    cases = [
        ("fixed", None, (5, 47, 19, 55)),
        ("central", None, (14, 23, 25, 34)),
        ("left", (6, 5, 38), (38, 5, 52, 13)),
        ("right", (6, 47, 47), (47, 47, 55, 55)),
    ]
    for name, selection, bbox in cases:
        for turns in range(4):
            node = env.clone()
            if selection:
                node.step(*selection)
            for _ in range(turns):
                node.step(5)
            colors = {
                int(v): int(n)
                for v, n in zip(*np.unique(node.frame(), return_counts=True))
            }
            print(name, "rot", turns, render_region(np.asarray(node.frame()), bbox),
                  colors)


harness.A.run_program("cn04", probe)
