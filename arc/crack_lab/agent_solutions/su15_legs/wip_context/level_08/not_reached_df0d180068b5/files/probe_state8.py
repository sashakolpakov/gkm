import json

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)

    def summary(node):
        frame = node.frame()
        top = tuple(
            (row, col, int(frame[row][col]))
            for row in range(10)
            for col in range(64)
            if int(frame[row][col]) == 7
        )
        return top, body_groups(frame)

    print("ROOT", summary(env))
    symbols = "0123456789ABCDEF"
    for row in range(10):
        print("UI", row, "".join(symbols[int(env.frame()[row][col])]
                                 for col in range(64)))
    node = env.clone()
    for index, action in enumerate(PREFIX, 1):
        node.step(*action)
        if index in (1, 5, 10, 15, len(PREFIX)):
            print("STEP", index, summary(node))

    top_blobs = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(node.frame(), min_area=1)
        if blob.bbox[2] < 10
    ]
    print("TOP_BLOBS", top_blobs)


H.A.run_program("su15", inspect)
