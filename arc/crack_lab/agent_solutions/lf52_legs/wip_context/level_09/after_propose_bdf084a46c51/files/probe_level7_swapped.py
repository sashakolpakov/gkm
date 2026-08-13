"""Replay level 7 with peg/bridge roles swapped through the carrier network."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import safe_step


def summary(node):
    _, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return {"level": node.levels_completed,
            "carriers": tuple(sorted(carriers)),
            "bridges": tuple(sorted(bridges)),
            "pegs": tuple(sorted(pegs))}


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:331]:
        safe_step(env, action)
    node = env.clone()

    # Carry the movable bridge down the left rail, then the peg down the
    # right rail, leaving their bottom-row roles swapped.
    play_actions(node, (3, 3, 1, 1, 4, 4, 4))
    play_lattice_moves(node, (((12, 42), (24, 42)),))
    play_actions(node, (3, 3, 3, 2, 2, 3, 3, 2))
    play_lattice_moves(node, (((42, 12), (54, 12)),))
    play_actions(node, (1, 4, 4, 1, 1, 3, 3, 3))
    play_lattice_moves(node, (((12, 6), (24, 6)),))
    play_actions(node, (4, 4, 4, 2, 2, 4, 4, 4, 2))
    play_lattice_moves(node, (((42, 42), (54, 42)),))
    print("SWAPPED_BOTTOM", summary(node))

    # The middle relay uses the same occupied positions; only piece roles
    # differ at each alternating jump.
    play_lattice_moves(node, (
        ((54, 12), (54, 24)), ((54, 24), (54, 36)),
        ((54, 36), (54, 48)), ((54, 42), (54, 54)),
        ((54, 4), (54, 16)),
    ))
    print("SWAPPED_RING", summary(node))
    play_actions(node, (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2))
    print("SWAPPED_ALIGN_10", summary(node))
    play_lattice_moves(node, (
        ((54, 10), (54, 22)), ((54, 16), (54, 28)),
    ))
    print("SWAPPED_11", summary(node))
    play_actions(node, (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4))
    play_lattice_moves(node, (((24, 34), (24, 46)),))
    play_actions(node, (3, 3, 3, 2, 2, 4, 4, 2))
    play_lattice_moves(node, (((54, 28), (42, 28)),))
    play_actions(node, (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2))
    play_lattice_moves(node, (
        ((18, 46), (30, 46)), ((24, 46), (36, 46)),
        ((30, 46), (42, 46)), ((36, 46), (36, 58)),
    ))
    print("SWAPPED_PARALLEL", summary(node))

    # Try the canonical final parallel sequence as a baseline.
    play_lattice_moves(node, (
        ((36, 18), (36, 30)), ((42, 6), (42, 18)),
        ((42, 18), (42, 30)),
    ))
    play_actions(node, (2, 3, 3, 3, 2, 2))
    play_lattice_moves(node, (((42, 30), (30, 30)),))
    play_actions(node, (1, 1, 4, 4, 4, 2, 4, 2))
    play_lattice_moves(node, (((24, 54), (36, 54)),))
    play_actions(node, (1, 3, 1, 3, 3, 2, 4, 2, 2))
    play_lattice_moves(node, (((42, 54), (30, 54)),))
    print("SWAPPED_FINAL", summary(node))


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
