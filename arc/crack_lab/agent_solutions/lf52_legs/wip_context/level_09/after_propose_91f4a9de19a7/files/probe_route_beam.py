"""Beam-search controller variants along a verified coordinate route."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_moves,
    _bridge_carrier_state,
    _movable_bridge_board,
    solve_compact_bridge_carrier_peg_solitaire,
)
from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.inner.step(action, *coordinates)


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    return groups


def key_candidates(keys, pair_delete):
    actions = (1, 2, 3, 4, 7) if os.environ.get("OPT_RESET") == "1" else (1, 2, 3, 4)
    out = {keys}
    for index, action in enumerate(keys):
        out.add(keys[:index] + keys[index + 1:])
        for replacement in actions:
            if replacement != action:
                out.add(keys[:index] + (replacement,) + keys[index + 1:])
        if index + 1 < len(keys) and keys[index] != keys[index + 1]:
            out.add(keys[:index] + (keys[index + 1], keys[index])
                    + keys[index + 2:])
    if pair_delete:
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                out.add(keys[:first] + keys[first + 1:second]
                        + keys[second + 1:])
    if os.environ.get("OPT_INSERT") == "1":
        for index in range(len(keys) + 1):
            for action in actions:
                out.add(keys[:index] + (action,) + keys[index:])
    if os.environ.get("OPT_INSERT2") == "1" and not keys:
        for first in actions:
            out.add((first,))
            for second in actions:
                out.add((first, second))
    return tuple(sorted(out, key=lambda path: (len(path), path)))


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def local_pattern(frame, clicks):
    image = arr(frame)
    first, second = clicks
    source = (first[2] - 1, first[1] - 1)
    destination = (second[2] - 1, second[1] - 1)
    midpoint = ((source[0] + destination[0]) // 2,
                (source[1] + destination[1]) // 2)
    return tuple(
        image[row:row + 4, col:col + 4].tobytes()
        for row, col in (source, midpoint, destination)
    )


def advance(node, clicks, expected, desired):
    before = frame_key(node)
    for action in clicks:
        safe_step(node, action)
    return (
        int(node.levels_completed) >= desired
        or (frame_key(node) != before
            and local_pattern(node.frame(), clicks) == expected)
    )


def movable_moves(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    occupied = pegs | bridges
    destinations = slots | carriers
    moves = []
    for kind, sources in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint in occupied
                    and destination in destinations
                    and destination not in occupied
                ):
                    moves.append((kind, source, destination))
    return tuple(sorted(set(moves) | set(_bridge_carrier_moves(frame))))


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    beam_width = int(os.environ.get("OPT_BEAM", "8"))
    pair_delete = os.environ.get("OPT_PAIR_DELETE") == "1"
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current
    if desired == 4:
        recorder = Recorder(entry.clone())
        solve_compact_bridge_carrier_peg_solitaire(recorder)
        groups = split(tuple(recorder.path))
    else:
        groups = split(campaign[start:end])
    if desired == 6:
        groups[11] = (
            (4,) * 9 + (1, 3, 3, 1, 1),
            groups[11][1],
        )
        if os.environ.get("OPT_L6_SHORT_EVENT") == "1":
            groups[11] = ((4,) * 7 + (1, 1), groups[11][1])
    if desired == 7:
        groups[21] = ((1, 1, 4, 4, 4, 2, 4, 2), groups[21][1])
        if os.environ.get("OPT_STAGE9_DETOUR") == "1":
            groups[9] = ((1,) + groups[9][0], groups[9][1])
        if os.environ.get("OPT_STAGE20_DETOUR") == "1":
            keys = groups[20][0]
            groups[20] = (keys[:2] + (1,) + keys[2:], groups[20][1])
        if os.environ.get("OPT_EARLY_PEG") == "1":
            order = (0, 1, 4, 5, 2, 3) + tuple(range(6, len(groups)))
            groups = [groups[index] for index in order]
        if os.environ.get("OPT_EARLY_PEG3") == "1":
            order = (0, 1, 4, 5, 6, 2, 3) + tuple(
                range(7, len(groups))
            )
            groups = [groups[index] for index in order]

    start_stage = int(os.environ.get("OPT_START_STAGE", "0"))
    root = entry.clone()
    prefix_path = ()
    for known_keys, clicks in groups[:start_stage]:
        for action in known_keys + clicks:
            safe_step(root, action)
        prefix_path += known_keys + clicks
    groups = groups[start_stage:]

    reference = root.clone()
    expected_patterns = []
    for known_keys, clicks in groups:
        for action in known_keys + clicks:
            safe_step(reference, action)
        expected_patterns.append(local_pattern(reference.frame(), clicks))

    frontier = [(len(prefix_path), root.clone(), prefix_path)]
    for stage, (known_keys, clicks) in enumerate(groups):
        candidates = key_candidates(known_keys, pair_delete)
        children = {}
        wins = []
        for cost, root, path in frontier:
            for keys in candidates:
                child = root.clone()
                for action in keys:
                    safe_step(child, action)
                if not advance(
                        child, clicks, expected_patterns[stage], desired):
                    continue
                child_cost = cost + len(keys) + 2
                child_path = path + keys + clicks
                if int(child.levels_completed) >= desired:
                    wins.append((child_cost, child_path))
                    continue
                child_key = frame_key(child)
                known = children.get(child_key)
                if known is None or child_cost < known[0]:
                    children[child_key] = (child_cost, child, child_path)
        if wins:
            best = min(wins)
            print("beam_win", desired, start_stage + stage,
                  best[0], best[1], flush=True)
            return
        frontier = sorted(children.values(), key=lambda item: item[0])[:beam_width]
        print("beam_stage", desired, start_stage + stage,
              len(candidates), len(children),
              tuple(cost for cost, _, _ in frontier), flush=True)
        if not frontier:
            return
        if start_stage + stage == int(os.environ.get("OPT_STOP_STAGE", "-1")):
            print("beam_stop", tuple(
                (cost, path, movable_moves(node.frame()))
                for cost, node, path in frontier
            ), flush=True)
            return
    print("beam_none", desired, flush=True)


arena.run_program("lf52", probe)
