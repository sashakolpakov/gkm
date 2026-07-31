"""Model-independent bounded configuration search near the level-6 start."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import POINTS, STARTS, covers


PIECES = "ABCD"


def safe_order(targets):
    for order in itertools.permutations(PIECES):
        if all(
            not any(
                covers(piece, *targets[piece], *POINTS[later])
                for later in order[index + 1:]
            )
            for index, piece in enumerate(order)
        ):
            return order
    return None


def axis_actions(delta, negative, positive):
    return [negative if delta < 0 else positive] * abs(delta)


def path_for(targets, order):
    path = []
    for piece in order:
        start_top, start_left = STARTS[piece]
        top, left = targets[piece]
        dy = (top - start_top) // 3
        dx = (left - start_left) // 3
        if not dx and not dy:
            continue
        path += [(6, *POINTS[piece])]
        path += axis_actions(dy, 1, 2)
        path += axis_actions(dx, 3, 4)
    return path + [5]


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def offset_vectors(radius):
    def generate(index, remaining, prefix):
        if index == 8:
            yield tuple(prefix)
            return
        for value in range(-remaining, remaining + 1):
            prefix.append(value)
            yield from generate(
                index + 1, remaining - abs(value), prefix,
            )
            prefix.pop()

    yield from generate(0, radius, [])


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    base_level = env.levels_completed
    tested = 0
    steps = 0
    seen = set()

    radius = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    exact_shell = "--shell" in sys.argv
    for offsets in offset_vectors(radius):
        if exact_shell and sum(abs(value) for value in offsets) != radius:
            continue
        targets = {}
        key = []
        valid = True
        for index, piece in enumerate(PIECES):
            start_top, start_left = STARTS[piece]
            top = start_top + 3 * offsets[2 * index]
            left = start_left + 3 * offsets[2 * index + 1]
            if not (5 <= left <= 56 and 14 <= top <= 56):
                valid = False
                break
            targets[piece] = (top, left)
            key.extend((top, left))
        if not valid or tuple(key) in seen:
            continue
        seen.add(tuple(key))
        order = safe_order(targets)
        if order is None:
            continue
        path = path_for(targets, order)
        result = replay(env, path)
        tested += 1
        steps += len(path)
        if result.levels_completed > base_level:
            print(
                "LOCAL_WIN", targets, "ORDER", order,
                "TESTED", tested, "STEPS", steps, "PATH", path,
            )
            return
        target_elapsed = steps / 280.0
        elapsed = time.monotonic() - started
        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)
    print("LOCAL_NONE", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
