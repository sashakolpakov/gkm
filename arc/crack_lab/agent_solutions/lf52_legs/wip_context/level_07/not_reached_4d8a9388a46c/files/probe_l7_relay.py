import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board, play_actions, play_lattice_moves
from perception import connected_components


def board(frame):
    return tuple(tuple(sorted(part)) for part in _movable_bridge_board(frame))


def setup(env, finish=True):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)

    play_actions(env, (3, 3, 1, 1, 3, 3, 3))
    play_lattice_moves(env, (((12, 6), (24, 6)),))
    play_actions(env, (4, 4, 4, 2, 2, 3, 3, 2))
    play_lattice_moves(env, (((42, 12), (54, 12)),))
    play_actions(env, (1, 4, 4, 1, 1, 4, 4, 4))
    play_lattice_moves(env, (((12, 42), (24, 42)),))
    play_actions(env, (3, 3, 3, 2, 2, 4, 4, 4, 2))
    play_lattice_moves(env, (((42, 42), (54, 42)),))
    play_lattice_moves(env, (
        ((54, 12), (54, 24)),
        ((54, 24), (54, 36)),
        ((54, 36), (54, 48)),
        ((54, 42), (54, 54)),
        ((54, 4), (54, 16)),
    ))
    play_actions(env, (3, 3, 3, 2, 2, 3, 2, 2, 4, 4, 2))
    play_lattice_moves(env, (
        ((54, 10), (54, 22)),
        ((54, 16), (54, 28)),
    ))
    play_actions(env, (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4))
    play_lattice_moves(env, (((24, 34), (24, 46)),))
    play_actions(env, (3, 3, 3, 2, 2, 4, 4, 2))
    play_lattice_moves(env, (((54, 28), (42, 28)),))
    play_actions(env, (1, 3, 3, 1, 1, 4, 4, 4, 3, 1, 1, 4, 4, 4, 2))
    if not finish:
        return
    play_lattice_moves(env, (
        ((18, 46), (30, 46)),
        ((24, 46), (36, 46)),
        ((36, 46), (36, 58)),
    ))


def probe(env):
    setup(env)
    play_lattice_moves(env, (
        ((36, 18), (36, 6)),
        ((30, 6), (42, 6)),
        ((42, 6), (42, 18)),
        ((42, 18), (42, 30)),
        ((36, 6), (36, 18)),
        ((36, 18), (36, 30)),
    ))
    print("ROOT", env.levels_completed, board(env.frame()))
    for color in (1, 3, 11, 12, 14, 15):
        print("COMPONENTS", color, [
            (blob.bbox, blob.size, blob.area)
            for blob in connected_components(env.frame(), colors=(color,))
            if blob.area < 100
        ])


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("RESULT", levels, len(path), error)
