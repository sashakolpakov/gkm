"""Prefix-sharing search over the protocol glyphs reproduced on level 4."""

import json
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A


ROWS = (33, 36, 39, 42, 45, 48)
COLS = (34, 39, 44, 49, 54, 59)
CODES = {
    "R": (1,),
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "X": (3,),
    "M": (0, 3),
}
SYMBOLS = tuple(sys.argv[1]) if len(sys.argv) > 1 else ("M", "L", "D", "X")


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base_level = env.levels_completed
    tested = 0
    steps = 0
    started = time.monotonic()

    def bounded_step(node, *action):
        nonlocal steps
        node.step(*action)
        steps += 1
        target_elapsed = steps / 280.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)

    def visit(node, depth, program):
        nonlocal tested
        if depth == len(COLS):
            tested += 1
            bounded_step(node, 6, 57, 58)
            if node.levels_completed > base_level:
                return program
            return None

        for symbol in SYMBOLS:
            child = node.clone()
            for row_index in CODES[symbol]:
                bounded_step(child, 6, COLS[depth], ROWS[row_index])
            result = visit(child, depth + 1, program + symbol)
            if result is not None:
                return result
        return None

    winner = visit(env.clone(), 0, "")
    print(
        "search",
        {
            "symbols": SYMBOLS,
            "tested": tested,
            "steps": steps,
            "winner": winner,
            "rate": round(steps / max(time.monotonic() - started, 0.001), 1),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
