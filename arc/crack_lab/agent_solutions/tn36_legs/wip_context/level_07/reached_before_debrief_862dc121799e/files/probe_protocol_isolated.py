"""Run one candidate protocol in an isolated harness process."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import connected_components


ROWS = (33, 36, 39, 42, 45, 48)
COLUMNS = (34, 39, 44, 49, 54, 59)
CODES = {
    "N": (),
    "A": (0,),
    "B": (0, 1, 5),
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "R": (1,),
    "X": (3,),
    "M": (0, 3),
}
PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else "MDNNNN"


def attempt(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for column_index, direction in enumerate(PROTOCOL):
        for row_index in CODES[direction]:
            env.step(6, COLUMNS[column_index], ROWS[row_index])

    submit = max(
        (
            blob
            for blob in connected_components(env.frame(), colors=(9,), min_area=4)
            if blob.size[0] > 1 and blob.size[1] > 1
        ),
        key=lambda blob: blob.area,
    )
    row, column = submit.centroid
    env.step(6, int(round(column)), int(round(row)))
    print("attempt", {"protocol": PROTOCOL, "levels": env.levels_completed})


levels, path, error = A.run_program("tn36", attempt)
valid = A.validate("tn36", path, levels) if path else False
print(
    "probe_result",
    {"levels": levels, "moves": len(path), "replay_ok": valid, "error": error},
)
