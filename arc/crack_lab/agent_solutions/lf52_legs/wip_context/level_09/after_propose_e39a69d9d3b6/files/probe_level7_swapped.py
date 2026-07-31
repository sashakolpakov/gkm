"""Verify the bridge-first, swapped-cargo level-7 route."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    with open("level7_greedy_macro_candidate.json") as native_file:
        native = json.load(native_file)
    for action in campaign[:331]:
        play(env, action)
    prefix = [
        3, 3, 1, 1, 4, 4, 4,
        [6, 43, 13], [6, 43, 25],
        3, 3, 3, 2, 2, 3, 3, 2,
        [6, 13, 43], [6, 13, 55],
        1, 4, 4, 1, 1, 3, 3, 3,
        [6, 7, 13], [6, 7, 25],
        3, 3, 3, 2, 2, 4, 4, 4, 2,
        [6, 43, 43], [6, 43, 55],
        [6, 13, 55], [6, 25, 55],
        [6, 25, 55], [6, 37, 55],
        [6, 37, 55], [6, 49, 55],
        [6, 43, 55], [6, 55, 55],
        [6, 5, 55], [6, 17, 55],
    ]
    candidate = prefix + native[50:]
    native_node = env.clone()
    swapped_node = env.clone()
    for action in native[:50]:
        play(native_node, action)
    for action in prefix:
        play(swapped_node, action)
    boundaries = {50, 65, 78, 88, 115, 123, 133, 144}
    print("COMPARE", {
        "at": 50,
        "native": _movable_bridge_board(native_node.frame())[1:],
        "swapped": _movable_bridge_board(swapped_node.frame())[1:],
    })
    for index, action in enumerate(native[50:], 51):
        play(native_node, action)
        play(swapped_node, action)
        if index in boundaries:
            print("COMPARE", {
                "at": index,
                "native_level": native_node.levels_completed,
                "swapped_level": swapped_node.levels_completed,
                "native": _movable_bridge_board(native_node.frame())[1:],
                "swapped": _movable_bridge_board(swapped_node.frame())[1:],
            })
    node = env.clone()
    executed = []
    for action in candidate:
        play(node, action)
        executed.append(action)
        if node.levels_completed > 6:
            break
    if node.levels_completed > 6:
        with open("level7_swapped_candidate.json", "w") as candidate_file:
            json.dump(executed, candidate_file, indent=2)
            candidate_file.write("\n")
    print("RESULT", {
        "prefix": len(prefix),
        "declared": len(candidate),
        "executed": len(executed),
        "level": node.levels_completed,
        "saved": 144 - len(executed) if node.levels_completed > 6 else None,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
