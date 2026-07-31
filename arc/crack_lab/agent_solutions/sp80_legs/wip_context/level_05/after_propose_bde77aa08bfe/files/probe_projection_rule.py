"""Symbolic comparison of known sp80 layouts and their border sockets."""
import importlib.util
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components


def load_players():
    spec = importlib.util.spec_from_file_location("players", "players.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(frame):
    keep = (4, 6, 8, 9, 11, 15)
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, colors=keep, min_area=4)
    )


PLANS = {
    2: (((14, 18), (4, 4, 4)), ((34, 26), (4, 4)),
        ((30, 38), (4, 4))),
    3: (((15, 21), (4,)), ((49, 29), (3,) * 10),
        ((19, 33), (4,) * 5), ((47, 41), (4,))),
    4: (((24, 18), (4, 4, 4)), ((45, 18), (3,) * 8),
        ((18, 30), (4,) * 8), ((49, 33), (3,) * 10),
        ((43, 42), (3,) * 7)),
}


def precommit(env, level):
    node = env.clone()
    selected = []
    for (x, y), moves in PLANS[level]:
        node.step(6, x, y)
        for action in moves:
            node.step(action)
        selected.append(tuple(
            (b.bbox, b.area)
            for b in connected_components(node.frame(), colors=(9,), min_area=4)
        ))
    node.step(5)
    return tuple(selected), node.levels_completed > env.levels_completed


def probe(env):
    players = load_players()
    for level in range(1, 5):
        print("START", level, summary(env.frame()))
        if level in PLANS:
            print("PRECOMMIT", level, precommit(env, level))
        players.__dict__[f"play_level_{level}"](env)
    print("START", 5, summary(env.frame()))


if __name__ == "__main__":
    levels, path, err = arena.run_program("sp80", probe)
    print("RESULT", levels, len(path), err)
