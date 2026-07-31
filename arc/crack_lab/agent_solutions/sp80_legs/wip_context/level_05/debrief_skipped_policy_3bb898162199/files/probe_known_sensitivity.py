"""One-variable perturbations around verified sp80 winning layouts."""
import importlib.util
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


def sensitivity(env, pieces, winning, values, step):
    results = []
    for varied in range(len(pieces)):
        accepted = []
        for candidate in values[varied]:
            targets = list(winning)
            targets[varied] = candidate
            node = env.clone()
            for (x, y, start), target in zip(pieces, targets):
                node.step(6, x, y)
                move_x(node, start, target, step)
            node.step(5)
            if node.levels_completed > env.levels_completed:
                accepted.append(candidate)
        results.append(tuple(accepted))
    return tuple(results)


def probe(env):
    players = load_players()
    players.play_level_1(env)
    print("SENS2", sensitivity(
        env,
        ((14, 18, 8), (34, 26, 28), (30, 38, 20)),
        (20, 36, 28),
        (tuple(range(0, 53, 4)),) * 3,
        4,
    ))
    players.play_level_2(env)
    print("SENS3", sensitivity(
        env,
        ((15, 21, 8), (49, 29, 40), (19, 33, 8), (47, 41, 36)),
        (12, 0, 28, 40),
        (tuple(range(0, 49, 4)), tuple(range(0, 45, 4)),
         tuple(range(0, 41, 4)), tuple(range(0, 41, 4))),
        4,
    ))
    players.play_level_3(env)
    print("SENS4", sensitivity(
        env,
        ((24, 18, 17), (45, 18, 38), (18, 30, 8),
         (49, 33, 44), (43, 42, 38)),
        (26, 14, 32, 14, 17),
        (tuple(range(5, 51, 3)),) * 5,
        3,
    ))


if __name__ == "__main__":
    levels, path, err = arena.run_program("sp80", probe)
    print("RESULT", levels, len(path), err)
