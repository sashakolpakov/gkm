import importlib.util
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import action_deltas, block_signatures, color_counts, connected_components


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        env.step(action)
    solver.solve(env)
    state = _bridge_carrier_state(env.frame())
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COLORS", color_counts(env.frame()))
    print("STATE", state)
    print("MOVES", _bridge_carrier_moves(env.frame()))
    print(
        "OBJECTS",
        [
            (b.color, b.bbox, b.area)
            for b in connected_components(
                env.frame(), colors=(1, 3, 11, 12, 14, 15), min_area=2
            )
        ],
    )
    print(
        "RAILS",
        [
            (b.color, b.bbox, b.area)
            for b in connected_components(
                env.frame(), colors=(5, 7, 9), min_area=2
            )
        ],
    )
    print(
        "DELTAS",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    symbols = {
        14: "P", 12: "C", 15: "B", 1: "o",
        7: "g", 11: "c", 9: "#", 5: "|",
    }
    signatures = block_signatures(env.frame(), cell=6)
    print("MAP")
    for row in range(11):
        print("".join(
            next(
                (symbols[color] for color in symbols if color in signatures[(row, col)]),
                ".",
            )
            for col in range(11)
        ))


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)
