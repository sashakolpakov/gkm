"""Small level-5 tests driven by projection-coverage hypotheses."""
import importlib.util
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components


def load_players():
    spec = importlib.util.spec_from_file_location("players", "players.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def translate(env, x, y, moves):
    env.step(6, x, y)
    for action in moves:
        env.step(action)


def runs(values):
    values = sorted(values)
    out = []
    for value in values:
        if not out or value != out[-1][1] + 1:
            out.append([value, value])
        else:
            out[-1][1] = value
    return tuple(map(tuple, out))


def projection(frame):
    f = arr(frame)
    mask = (f == 8) | (f == 9) | (f == 15)
    ys, xs = mask.nonzero()
    blobs = tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, colors=(8, 9, 15), min_area=4)
    )
    return runs(set(map(int, xs))), runs(set(map(int, ys))), blobs


def try_plan(env, name, plan):
    node = env.clone()
    for x, y, moves in plan:
        translate(node, x, y, moves)
    before = projection(node.frame())
    node.step(5)
    print("TRY", name, node.levels_completed > env.levels_completed, before)


def probe(env):
    players = load_players()
    for level in range(1, 5):
        players.__dict__[f"play_level_{level}"](env)
    print("BASE", projection(env.frame()))
    try_plan(env, "commit", ())
    try_plan(env, "D35_B41", (
        (34, 43, (1, 1)),
        (24, 33, (2, 2, 2)),
    ))
    try_plan(env, "D35_A41", (
        (34, 43, (1, 1)),
        (34, 21, (2,) * 7),
    ))
    try_plan(env, "bars35_38_41", (
        (24, 33, (2,)),
        (48, 33, (2, 2)),
        (34, 21, (2,) * 7),
    ))


if __name__ == "__main__":
    levels, path, err = arena.run_program("sp80", probe)
    print("RESULT", levels, len(path), err)
