import json

import gkm_try as H

from probe_clean8 import body_groups


def centers(frame):
    return tuple(
        (
            round(sum(row for row, _ in group) / len(group)),
            round(sum(col for _, col in group) / len(group)),
        )
        for group in body_groups(frame)
    )


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    base_groups = body_groups(env.frame())
    before = centers(env.frame())
    print("CENTERS", before)
    for body_index, group in enumerate(base_groups):
        center_row, center_col = before[body_index]
        for row, col in group:
            clone = env.clone()
            clone.step(6, col, row)
            after = centers(clone.frame())
            delta = tuple(
                (ar - br, ac - bc)
                for (ar, ac), (br, bc) in zip(after, before)
            )
            print(
                "CONTROL",
                body_index,
                (row - center_row, col - center_col),
                delta,
            )
    clone = env.clone()
    clone.step(6, 0, 10)
    after = centers(clone.frame())
    print(
        "BLANK",
        tuple(
            (ar - br, ac - bc)
            for (ar, ac), (br, bc) in zip(after, before)
        ),
    )


H.A.run_program("su15", inspect)
