"""Inspect public reset states immediately after L9's first reveal."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (_bridge_carrier_moves, _bridge_carrier_state,
                  _movable_bridge_board)
from perception import safe_step


OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def compact(node):
    state = _bridge_carrier_state(node.frame())
    movable = _movable_bridge_board(node.frame())
    return (
        len(state[0]), tuple(sorted(state[1])),
        tuple(sorted(state[2])), tuple(sorted(state[3])),
        tuple(sorted(state[4])), state[5],
        tuple(sorted(movable[1])), tuple(sorted(movable[2])),
        tuple(sorted(movable[3])),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in OPENING:
        move(env, source, destination)
    for _ in range(9):
        safe_step(env, 4)

    for count in range(16):
        print("frontier_undo_state", count, int(env.levels_completed),
              compact(env), _bridge_carrier_moves(env.frame()), flush=True)
        safe_step(env, 7)


arena.run_program("lf52", probe)
