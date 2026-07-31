"""Small black-box constraint probes against known winning arrangements."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3):
        player(env)

    l4_arranged = apply(env, (
        [(6, 24, 18)] + [4] * 3
        + [(6, 45, 18)] + [3] * 8
        + [(6, 18, 30)] + [4] * 8
        + [(6, 49, 33)] + [3] * 10
        + [(6, 43, 42)] + [3] * 7
    ))
    l4_pieces = {
        "A15": (30, 18),
        "B15": (20, 18),
        "C21": (36, 30),
        "D12": (20, 33),
        "E12": (22, 42),
    }
    print("L4_PERTURB", {
        (name, direction): apply(
            l4_arranged, [(6, *point), direction, 5]
        ).levels_completed
        for name, point in l4_pieces.items()
        for direction in (1, 2, 3, 4)
    })

    play_level_4(env)
    arranged = apply(env, (
        [(6, 48, 33)] + [3] * 6
        + [(6, 21, 33)] + [2] * 3
        + [(6, 34, 21)] + [4] * 3 + [2] * 7
        + [(6, 36, 45)] + [3] * 6 + [1] * 2
    ))
    pieces = {
        "wide15": (30, 33),
        "wide9": (22, 42),
        "wide12": (42, 42),
        "L": (17, 37),
    }
    checks = {}
    for name, point in pieces.items():
        for direction in (1, 2, 3, 4):
            node = apply(arranged, [(6, *point), direction, 5])
            checks[(name, direction)] = node.levels_completed
    checks[("baseline", 5)] = apply(arranged, [5]).levels_completed
    print("L5_PERTURB", checks)
    l5_ranges = {}
    for name, point in pieces.items():
        for direction in (1, 2, 3, 4):
            wins = []
            for count in range(7):
                node = apply(
                    arranged,
                    [(6, *point)] + [direction] * count + [5],
                )
                if node.levels_completed > env.levels_completed:
                    wins.append(count)
            l5_ranges[(name, direction)] = tuple(wins)
    print("L5_RANGES", l5_ranges)

    play_level_5(env)
    common = (
        [(6, 30, 19)] + [2] * 3
        + [(6, 45, 18)] + [2] * 3
        + [(6, 33, 45)] + [1] * 3
    )
    candidates = {
        "chain_B38": common + [(6, 40, 25)] + [3] * 2 + [5],
        "chain_B20": common + [(6, 40, 25)] + [3] * 8 + [5],
        "A23": (
            [(6, 30, 19)] + [2] * 2
            + [(6, 45, 18)] + [2] * 3 + [3] * 2
            + [(6, 33, 45)] + [1] * 3 + [5]
        ),
        "A29": (
            [(6, 30, 19)] + [2] * 4
            + [(6, 45, 18)] + [2] * 3 + [3] * 2
            + [(6, 33, 45)] + [1] * 3 + [5]
        ),
        "B20_A26_D35": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 45, 18)] + [2] * 2 + [3] * 2
            + [(6, 33, 45)] + [1] * 3 + [5]
        ),
        "B26_A26_D35": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 45, 18)] + [2] * 4 + [3] * 2
            + [(6, 33, 45)] + [1] * 3 + [5]
        ),
        "D38": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 45, 18)] + [2] * 3 + [3] * 2
            + [(6, 33, 45)] + [1] * 2 + [5]
        ),
    }
    print("L6_CANDIDATES", {
        name: apply(env, path).levels_completed
        for name, path in candidates.items()
    })


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
