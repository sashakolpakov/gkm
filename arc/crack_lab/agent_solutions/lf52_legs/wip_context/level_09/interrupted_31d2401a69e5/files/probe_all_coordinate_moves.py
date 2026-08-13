"""Brute-validate every visible piece-to-destination coordinate pair."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def patches(frame):
    array = np.asarray(frame)
    windows = np.lib.stride_tricks.sliding_window_view(array, (4, 4))
    counts = {
        color: np.count_nonzero(windows == color, axis=(-1, -2))
        for color in (1, 8, 9, 12, 14)
    }

    def positions(mask):
        rows, cols = np.where(mask)
        return set(zip(map(int, rows), map(int, cols)))

    destinations = positions(counts[1] == 16) | positions(counts[12] == 16)
    pieces = {
        8: positions(counts[8] >= 12),
        9: positions(counts[9] >= 12),
        14: positions(counts[14] >= 12),
    }
    bridge_state = _bridge_carrier_state(frame)
    destinations |= set(bridge_state[0]) | set(bridge_state[2])
    pieces[14] |= set(bridge_state[1])
    movable = _movable_bridge_board(frame)
    destinations |= set(movable[0]) | set(movable[1])
    pieces[8] |= set(movable[2])
    pieces[14] |= set(movable[3])
    return destinations, pieces


def play(env, action):
    safe_step(env, tuple(action) if isinstance(action, list) else action)


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "9"))
    context = int(os.environ.get("CONTEXT_ACTIONS", "0"))
    candidate_name = os.environ.get("CANDIDATE")
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:BOUNDARIES[level - 1]]:
        play(env, action)
    root = env.clone()
    if candidate_name:
        with open(candidate_name) as stream:
            candidate = json.load(stream)
        for action in candidate[:context]:
            play(root, action)

    base_level = int(root.levels_completed)
    destinations, pieces = patches(root.frame())
    successes = []
    tested = 0
    for color, sources in pieces.items():
        for source in sorted(sources):
            for destination in sorted(destinations - {source}):
                tested += 1
                child = root.clone()
                safe_step(child, (6, source[1] + 1, source[0] + 1))
                safe_step(
                    child,
                    (6, destination[1] + 1, destination[0] + 1),
                )
                if child.levels_completed > base_level:
                    successes.append((color, source, destination, "reward"))
                    continue
                _, after_pieces = patches(child.frame())
                if (
                    source not in after_pieces[color]
                    and destination in after_pieces[color]
                ):
                    successes.append((
                        color,
                        source,
                        destination,
                        (
                            abs(source[0] - destination[0]),
                            abs(source[1] - destination[1]),
                        ),
                    ))
    print("ALL_COORDINATE_MOVES", {
        "level": level,
        "context": context,
        "tested": tested,
        "destinations": len(destinations),
        "pieces": {color: sorted(value) for color, value in pieces.items()},
        "successes": successes,
    })


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {
    "levels": levels,
    "moves": len(path),
    "error": str(error),
})
