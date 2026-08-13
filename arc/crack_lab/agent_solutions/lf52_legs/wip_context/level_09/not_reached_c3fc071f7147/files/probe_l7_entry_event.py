"""Test the legal level-7 entry event against the verified complete suffix."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_parallel_wrapped_bridge_carrier_peg_solitaire
from perception import arr, safe_step
from probe_key_neighborhood_events import generic_moves
from legs import _bridge_carrier_state
from perception import connected_components


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


def flatten(groups):
    return tuple(action for keys, clicks in groups for action in keys + clicks)


def replay(entry, path):
    node = entry.clone()
    changed = []
    index = 0
    while index < len(path):
        action = path[index]
        if isinstance(action, int):
            safe_step(node, action)
            index += 1
            continue
        before = arr(node.frame())[1:, :].tobytes()
        safe_step(node, action)
        safe_step(node, path[index + 1])
        changed.append(arr(node.frame())[1:, :].tobytes() != before)
        index += 2
    return node, tuple(changed)


def key_tree_fresh(root, depth):
    states = {}
    paths = [()]
    for _ in range(depth + 1):
        next_paths = []
        for path in paths:
            node = root.clone()
            for action in path:
                safe_step(node, action)
            states.setdefault(arr(node.frame())[1:, :].tobytes(), path)
            next_paths.extend(path + (action,) for action in (1, 2, 3, 4))
        paths = next_paths
    return states


def key_tree_undo(root, depth, undo_noops):
    node = root.clone()
    states = {}
    failed = []

    def visit(path):
        states.setdefault(arr(node.frame())[1:, :].tobytes(), path)
        if len(path) >= depth:
            return
        for action in (1, 2, 3, 4):
            before = arr(node.frame())[1:, :].tobytes()
            safe_step(node, action)
            changed = arr(node.frame())[1:, :].tobytes() != before
            visit(path + (action,))
            if changed or undo_noops:
                safe_step(node, 7)
            if arr(node.frame())[1:, :].tobytes() != before:
                failed.append((path, action, changed))
                return

    visit(())
    return states, failed


def probe(env):
    with open("frontier_scaffold.json") as stream:
        prefix = json.load(stream)["staged_prefix_actions"]
    for action in prefix[:318]:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    entry = env.clone()

    recorder = Recorder(entry.clone())
    solve_parallel_wrapped_bridge_carrier_peg_solitaire(recorder)
    groups = split(tuple(recorder.path))
    pieces = tuple(
        (blob.color, blob.top_left, blob.size, blob.area)
        for blob in connected_components(entry.frame(), colors=(8, 9, 12, 14, 15))
        if blob.size in ((2, 2), (4, 4))
    )
    print("entry", int(entry.levels_completed), pieces,
          _bridge_carrier_state(entry.frame()), flush=True)
    print("entry_moves", generic_moves(entry.frame()), flush=True)
    undo_checks = []
    for action in (1, 2, 3, 4):
        child = entry.clone()
        before = arr(child.frame())[1:, :].tobytes()
        safe_step(child, action)
        changed = arr(child.frame())[1:, :].tobytes() != before
        safe_step(child, 7)
        undo_checks.append((action, changed,
                            arr(child.frame())[1:, :].tobytes() == before))
    print("entry_undo", tuple(undo_checks), flush=True)
    fresh = key_tree_fresh(entry, 3)
    for undo_noops in (False, True):
        states, failed = key_tree_undo(entry, 3, undo_noops)
        print("key_tree", undo_noops, len(fresh), len(states),
              len(set(fresh) - set(states)), tuple(failed), flush=True)

    for move in generic_moves(entry.frame()):
        _, source, destination = move
        clicks = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
        candidate = clicks + flatten(groups)
        node, changed = replay(entry, candidate)
        print("entry_event", move, len(candidate), int(node.levels_completed),
              tuple(index for index, value in enumerate(changed) if not value),
              flush=True)
        for omitted in range(len(groups)):
            shorter = clicks + flatten(groups[:omitted] + groups[omitted + 1:])
            child, _ = replay(entry, shorter)
            if int(child.levels_completed) >= 7:
                print("entry_win", move, omitted, len(shorter), shorter,
                      flush=True)


arena.run_program("lf52", probe)
