"""Test geometry-driven level-4 programs on pristine clones."""

import json
import sys
from collections import Counter
from itertools import permutations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components, frame_delta


ROWS = (33, 36, 39, 42, 45, 48)
COLS = (34, 39, 44, 49, 54, 59)
CODES = {
    "N": (),
    "R": (1,),
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "X": (3,),
    "M": (0, 3),
}


def board_objects(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(11,))
        if blob.bbox[2] < 32 and blob.bbox[1] > 31
    )


def apply_program(node, program):
    for col, symbol in zip(COLS, program):
        for row_index in CODES[symbol]:
            node.step(6, col, ROWS[row_index])


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    programs = set()
    for special in ("X", "M"):
        for symbols in (
            special + "L" + "DDDD",
            special + "LL" + "DDD",
            special + "L" + "DDD" + "N",
        ):
            programs.update("".join(order) for order in set(permutations(symbols)))
    for symbols in (
        "XMLDDD",
        "XMLLDD",
        "XMLLLD",
        "XMDDDD",
    ):
        programs.update("".join(order) for order in set(permutations(symbols)))

    outcomes = Counter()
    wins = []
    for program in programs:
        clone = env.clone()
        apply_program(clone, program)
        before = arr(clone.frame()).copy()
        clone.step(6, 57, 58)
        after = arr(clone.frame()).copy()
        delta = frame_delta(before, after)
        result = (
            clone.levels_completed - env.levels_completed,
            delta["count"],
            delta["bbox"],
            board_objects(after),
        )
        outcomes[result] += 1
        if clone.levels_completed > env.levels_completed:
            wins.append((program, result))
    print("tested", len(programs))
    print("wins", wins)
    for outcome, count in outcomes.most_common():
        print("outcome", count, outcome)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
