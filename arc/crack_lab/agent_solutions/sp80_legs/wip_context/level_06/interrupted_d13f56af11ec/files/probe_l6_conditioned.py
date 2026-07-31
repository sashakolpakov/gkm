"""Conditioned level-6 search over socket-matched placements."""
import random
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


STARTS = {"A": (17, 29), "B": (14, 44), "C": (32, 23), "D": (44, 29)}
POINTS = {"A": (30, 19), "B": (45, 18), "C": (25, 33), "D": (33, 45)}
ORDER = ("C", "A", "D", "B")
SIDE_OUTER = (20, 29, 35)
SIDE_GAP = (23, 32, 38)


def axis_moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def covers(piece, top, left, x, y):
    row, col = y - top, x - left
    if piece == "A":
        return (
            (0 <= row < 3 and 0 <= col < 3)
            or (3 <= row < 6 and 0 <= col < 6)
        )
    if piece == "D":
        return (
            (0 <= row < 3 and 3 <= col < 6)
            or (3 <= row < 6 and 0 <= col < 6)
        )
    height, width = ((12, 3) if piece == "B" else (3, 15))
    return 0 <= row < height and 0 <= col < width


def safe_targets(targets):
    for index, piece in enumerate(ORDER):
        top, left = targets[piece]
        for later in ORDER[index + 1:]:
            if covers(piece, top, left, *POINTS[later]):
                return False
    return True


def path_for(targets):
    path = []
    for piece in ORDER:
        start_top, start_left = STARTS[piece]
        top, left = targets[piece]
        path += [(6, *POINTS[piece])]
        path += axis_moves(start_top, top, 1, 2)
        path += axis_moves(start_left, left, 3, 4)
    return path + [5]


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    rng = random.Random(806806)
    base_level = env.levels_completed
    started = time.monotonic()
    steps = 0
    tested = 0

    while tested < 800:
        pieces = list(STARTS)
        rng.shuffle(pieces)
        bottom_piece, side_pieces = pieces[0], pieces[1:]
        targets = {}
        for socket_index, piece in enumerate(side_pieces):
            if piece in ("A", "D"):
                top = SIDE_OUTER[socket_index]
                left = rng.choice(tuple(range(5, 54, 3)))
            elif piece == "B":
                gap = SIDE_GAP[socket_index]
                top = rng.choice(tuple(
                    value for value in range(14, 45, 3)
                    if value <= gap <= value + 9
                ))
                left = rng.choice(tuple(range(5, 57, 3)))
            else:
                top = SIDE_GAP[socket_index]
                left = rng.choice(tuple(range(5, 45, 3)))
            targets[piece] = (top, left)

        if bottom_piece in ("A", "D"):
            left = rng.choice((26, 29))
            top = rng.choice(tuple(range(14, 51, 3)))
        elif bottom_piece == "B":
            left = 29
            top = rng.choice(tuple(range(14, 48, 3)))
        else:
            left = rng.choice((17, 20, 23, 26, 29))
            top = rng.choice(tuple(range(14, 54, 3)))
        targets[bottom_piece] = (top, left)
        if not safe_targets(targets):
            continue

        path = path_for(targets)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "CONDITIONED_WIN", targets, "TESTED", tested,
                "STEPS", steps, "PATH", path,
            )
            return
        target_elapsed = steps / 280.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print("CONDITIONED_NONE", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
