"""Enumerate cargo events near one verified carrier-key alignment."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state, solve_compact_bridge_carrier_peg_solitaire
from perception import arr, connected_components, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


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
    return tuple(groups)


def generic_moves(frame):
    blobs = connected_components(frame, colors=(1, 8, 9, 12, 14, 15))
    slots = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    slots |= {
        (blob.bbox[0] - 1, blob.bbox[1] - 1) for blob in blobs
        if blob.color == 1 and blob.size == (2, 2) and blob.area == 4
    }
    carriers = {
        ((blob.bbox[0] - 1, blob.bbox[1] - 1)
         if blob.size == (2, 2) else blob.top_left)
        for blob in blobs
        if blob.color == 12 and blob.size in ((2, 2), (4, 4))
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    movable = {
        blob.top_left for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    leapfrog = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    destinations = slots | carriers
    occupied = pegs | movable | leapfrog
    moves = []
    for kind, sources in (
            ("peg", pegs), ("movable", movable), ("leapfrog", leapfrog)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if destination not in destinations or destination in occupied:
                    continue
                legal = midpoint in occupied | fixed
                if legal:
                    moves.append((kind, source, destination))
    return tuple(sorted(moves))


def candidates(keys):
    out = {keys}
    for index, action in enumerate(keys):
        out.add(keys[:index] + keys[index + 1:])
        for replacement in (1, 2, 3, 4, 7):
            if replacement != action:
                out.add(keys[:index] + (replacement,) + keys[index + 1:])
        if index + 1 < len(keys) and keys[index] != keys[index + 1]:
            out.add(keys[:index] + (keys[index + 1], keys[index])
                    + keys[index + 2:])
    if os.environ.get("OPT_PAIR_DELETE") == "1":
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                out.add(keys[:first] + keys[first + 1:second]
                        + keys[second + 1:])
    return tuple(sorted(out, key=lambda path: (len(path), path)))


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


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    stage = int(os.environ.get("OPT_STAGE", "9"))
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
    if desired == 4 and os.environ.get("OPT_COMPACT") == "1":
        recorder = Recorder(entry.clone())
        solve_compact_bridge_carrier_peg_solitaire(recorder)
        groups = split(tuple(recorder.path))
    else:
        groups = split(campaign[start:end])
    root = entry
    for keys, clicks in groups[:stage]:
        for action in keys + clicks:
            safe_step(root, action)
    known, clicks = groups[stage]
    outcomes = {}
    for path in candidates(known):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        state_key = arr(node.frame())[1:, :].tobytes()
        if state_key in outcomes:
            continue
        moves = generic_moves(node.frame())
        if moves:
            outcomes[state_key] = (len(path), path, moves)
    print("neighborhood", desired, stage, known, clicks,
          len(candidates(known)),
          tuple(sorted(outcomes.values())), flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
