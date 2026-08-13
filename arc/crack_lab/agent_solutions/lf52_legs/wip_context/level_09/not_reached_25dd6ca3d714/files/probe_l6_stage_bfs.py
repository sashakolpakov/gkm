"""Apply the exact visible movable-bridge BFS at a level-6 relay stage."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_solution, play_lattice_moves
from perception import safe_step


def observe(env):
    stage = int(os.environ.get("AFTER_MACROS", "15"))
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:238]:
        safe_step(env, action)
    coordinate_actions = 0
    for action in path[238:331]:
        if coordinate_actions >= 2 * stage:
            break
        safe_step(env, action)
        if isinstance(action, list):
            coordinate_actions += 1
    solution = _movable_bridge_solution(env.frame(), max_states=200000)
    print("STAGE_BFS", {"after_macros": stage,
                        "macros": None if solution is None else len(solution),
                        "solution": solution})
    if solution is not None:
        node = env.clone()
        play_lattice_moves(node, solution)
        print("STAGE_REPLAY", {"level": node.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
