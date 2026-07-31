"""Sample alternative winning column layouts in known sp80 levels."""
import importlib.util
import random
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


def load_players():
    spec = importlib.util.spec_from_file_location("players", "players.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def move_x(env, start, target, step):
    action = 3 if target < start else 4
    for _ in range(abs(target - start) // step):
        env.step(action)


def sample(env, pieces, values, step, trials, seed):
    rng = random.Random(seed)
    wins = set()
    for _ in range(trials):
        node = env.clone()
        choice = tuple(rng.choice(values[i]) for i in range(len(pieces)))
        for (x, y, start), target in zip(pieces, choice):
            node.step(6, x, y)
            move_x(node, start, target, step)
        node.step(5)
        if node.levels_completed > env.levels_completed:
            wins.add(choice)
    return tuple(sorted(wins))


def probe(env):
    players = load_players()
    players.play_level_1(env)
    wins2 = sample(
        env,
        ((14, 18, 8), (34, 26, 28), (30, 38, 20)),
        (tuple(range(0, 53, 4)),) * 3,
        4, 0, 802,
    )
    print("WINS2", wins2)
    players.play_level_2(env)
    wins3 = sample(
        env,
        ((15, 21, 8), (49, 29, 40), (19, 33, 8), (47, 41, 36)),
        (tuple(range(0, 49, 4)), tuple(range(0, 45, 4)),
         tuple(range(0, 41, 4)), tuple(range(0, 41, 4))),
        4, 0, 803,
    )
    print("WINS3", wins3)
    node = env.clone()
    node.step(6, 15, 21)
    node.step(4)
    node.step(6, 19, 33)
    for _ in range(5):
        node.step(4)
    node.step(5)
    print("COVER3", node.levels_completed > env.levels_completed)


if __name__ == "__main__":
    levels, path, err = arena.run_program("sp80", probe)
    print("RESULT", levels, len(path), err)
