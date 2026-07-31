"""Exhaust representative piece-to-port assignments for sp80 level 6."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import POINTS, STARTS, covers


SIDE_GAPS = {"UL": 23, "R": 32, "LL": 38}
SIDE_OUTER = {"UL": 20, "R": 29, "LL": 35}
BOTTOM_LEFT = {"A": 29, "B": 29, "C": 23, "D": 26}
BOTTOM_TOPS = {
    "A": tuple(range(14, 54, 3)),
    "B": tuple(range(14, 48, 3)),
    "C": tuple(range(14, 57, 3)),
    "D": tuple(range(14, 54, 3)),
}
MODE = "central" if "--central" in sys.argv else "edges"


def side_top(piece, port):
    if piece in ("A", "D"):
        return (SIDE_OUTER[port],)
    if piece == "C":
        return (SIDE_GAPS[port],)
    gap = SIDE_GAPS[port]
    return tuple(
        top for top in range(14, 48, 3)
        if top <= gap <= top + 11
    )


def side_left(piece, port):
    if MODE == "central":
        return {"A": 29, "B": 29, "C": 23, "D": 26}[piece]
    if port in ("UL", "LL"):
        return 14
    if piece in ("A", "D"):
        return 47
    if piece == "C":
        return 38
    return 50


def moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def safe_order(targets):
    for order in itertools.permutations("ABCD"):
        safe = True
        for index, piece in enumerate(order):
            top, left = targets[piece]
            if any(
                covers(piece, top, left, *POINTS[later])
                for later in order[index + 1:]
            ):
                safe = False
                break
        if safe:
            return order
    return None


def path_for(targets, order):
    path = []
    for piece in order:
        start_top, start_left = STARTS[piece]
        top, left = targets[piece]
        path += [(6, *POINTS[piece])]
        path += moves(start_top, top, 1, 2)
        path += moves(start_left, left, 3, 4)
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
    started = time.monotonic()
    base_level = env.levels_completed
    tested = 0
    steps = 0

    for bottom_piece in "ABCD":
        side_pieces = tuple(piece for piece in "ABCD" if piece != bottom_piece)
        for assigned in itertools.permutations(side_pieces):
            roles = dict(zip(("UL", "R", "LL"), assigned))
            top_choices = tuple(
                side_top(piece, port) for port, piece in roles.items()
            )
            for side_tops in itertools.product(*top_choices):
                base_targets = {
                    piece: (
                        side_tops[index],
                        side_left(piece, port),
                    )
                    for index, (port, piece) in enumerate(roles.items())
                }
                for bottom_top in BOTTOM_TOPS[bottom_piece]:
                    targets = dict(base_targets)
                    targets[bottom_piece] = (
                        bottom_top, BOTTOM_LEFT[bottom_piece],
                    )
                    order = safe_order(targets)
                    if order is None:
                        continue
                    path = path_for(targets, order)
                    result = replay(env, path)
                    tested += 1
                    steps += len(path)
                    if result.levels_completed > base_level:
                        print(
                            "ROLE_WIN", targets, "ORDER", order,
                            "TESTED", tested, "STEPS", steps, "PATH", path,
                        )
                        return
                    target_elapsed = steps / 280.0
                    elapsed = time.monotonic() - started
                    if target_elapsed > elapsed:
                        time.sleep(target_elapsed - elapsed)
    print("ROLE_NONE", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
