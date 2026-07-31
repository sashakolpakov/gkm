"""Replay the strongest preserved level-9 routes from a pristine clone."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components


class ReachedLevel9(Exception):
    pass


class StopAtLevel9:
    def __init__(self, env):
        self.env = env
        self.moves = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        self.moves += 1
        if self.env.levels_completed >= 8:
            raise ReachedLevel9
        return result


TARGET = {
    (3, 5), (3, 6), (3, 7),
    (4, 5),         (4, 7),
    (5, 5), (5, 6), (5, 7),
}


def boxes(frame, color):
    return tuple(
        blob.bbox
        for blob in connected_components(frame, colors=(color,), min_area=4)
    )


def target_state(frame):
    grid = np.asarray(frame)
    empty = []
    filled = []
    for row, col in sorted(TARGET):
        colors = set(int(v) for v in grid[
            row * 4:row * 4 + 4,
            col * 4:col * 4 + 4,
        ].flat)
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        if 4 in colors and 9 in colors:
            filled.append((row, col))
    return tuple(empty), tuple(filled)


def tile_map(frame):
    grid = np.asarray(frame)
    rows = []
    for row in range(16):
        glyphs = []
        for col in range(16):
            colors = set(int(v) for v in grid[
                row * 4:row * 4 + 4,
                col * 4:col * 4 + 4,
            ].flat)
            symbol = "."
            for color, candidate in (
                (14, "A"), (15, "X"), (12, "C"), (4, "o")
            ):
                if color in colors:
                    symbol = candidate
                    break
            else:
                if colors == {5}:
                    symbol = "#"
                elif 5 in colors:
                    symbol = "+"
                elif 2 in colors and 9 in colors:
                    symbol = "T"
                elif 2 in colors:
                    symbol = "="
                elif 7 in colors:
                    symbol = "|"
            glyphs.append(symbol)
        rows.append("".join(glyphs))
    return tuple(rows)


def direct_second_prefix():
    remote_pick = [2] + [4] * 6 + [1, 5]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = first_delivery + [2] * 2 + [4] * 5 + [1, 5]
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    return second_pick + place_bottom_middle


COMBINED_DISMISS_PICK = [2, 2, 3, 5, 3, 3, 3]
POSITION = [4] + [1] * 6 + [4] * 4
EXACT_SUFFIX = [2, 4, 5, 1, 3, 2, 5, 2, 5, 1]
SHORT_SUFFIX = [2, 4, 5, 1, 3, 2, 2, 5, 1, 5]


def current_route():
    return (
        [2] + [4] * 6 + [1, 5] + [1] * 2 + [5, 2]
        + [3] * 2 + [1, 5, 4, 1, 5, 2]
        + [3] * 2 + [1] * 3 + [5] + [1] * 2 + [5, 2]
        + [2] * 3 + [3] * 5 + [2] * 2 + [3, 5]
        + [3] * 4 + [1, 5] + [4] * 7 + [1] * 3 + [3, 5]
    )


def replay(base, name, route):
    child = base.clone()
    base_level = child.levels_completed
    prior = target_state(child.frame())
    events = []
    for turn, action in enumerate(route, 1):
        child.step(action)
        current = target_state(child.frame())
        if current != prior or turn >= 63:
            events.append((turn, action, current))
        prior = current
        if child.terminal() or child.levels_completed > base_level:
            break
    print(
        "ROUTE",
        name,
        {
            "requested": len(route),
            "reward": child.levels_completed - base_level,
            "terminal": child.terminal(),
            "events": tuple(events),
            "avatar": boxes(child.frame(), 14),
            "couriers": boxes(child.frame(), 12),
            "thief": boxes(child.frame(), 15),
            "cargo": boxes(child.frame(), 4),
            "target": target_state(child.frame()),
        },
        flush=True,
    )


def inspect(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    transitions = []
    for index, action in enumerate(checkpoint["final_path"], 1):
        before = env.levels_completed
        env.step(action)
        if env.levels_completed != before:
            transitions.append((env.levels_completed, index))
        if env.levels_completed >= 8:
            break
    print(
        "PRISTINE",
        {"prior_moves": index, "transitions": tuple(transitions)},
        flush=True,
    )
    print("MAP", *tile_map(env.frame()), sep="\n", flush=True)
    prefix = direct_second_prefix() + COMBINED_DISMISS_PICK + [5] + POSITION
    replay(env, "current", current_route() + [5] * 9)
    replay(env, "frontier_exact", prefix + EXACT_SUFFIX)
    replay(env, "frontier_short", prefix + SHORT_SUFFIX)


if __name__ == "__main__":
    arena.run_program("wa30", inspect)
