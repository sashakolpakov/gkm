"""Measure hidden rail effects after undoing and redoing the phase transition."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level9 import pieces, symbolic_solution


SOURCE = (6, 31, 37)
DESTINATION = (6, 43, 37)


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    state = pieces(node.frame())
    return state["pegs"], state["carriers"], state["bridges"], state["fixed_bridges"]


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    solution, _ = symbolic_solution(env.frame())
    for _, source, destination in solution:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))
    loaded = env.clone()

    cycle = (7, 7, SOURCE, DESTINATION)
    setups = tuple((f"cycles_{count}", cycle * count) for count in range(7))
    baseline = {}
    base = loaded.clone()
    for count in range(11):
        baseline[count] = frame_key(base)
        safe_step(base, 4)

    for label, setup in setups:
        node = loaded.clone()
        for action in setup:
            safe_step(node, action)
        rows = []
        for count in range(11):
            rows.append((count, frame_key(node) == baseline[count], compact(node)))
            safe_step(node, 4)
        print("COHERENCE_CYCLE", label, len(setup), tuple(rows))


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
