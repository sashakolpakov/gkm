"""Beam over level-7 physical events with reversible bounded key closures."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_parallel_wrapped_bridge_carrier_peg_solitaire
from perception import arr, connected_components, safe_step
from probe_key_neighborhood_events import generic_moves


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


def frame_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def clicks(move):
    _, source, destination = move
    return (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    )


def phase(frame):
    arrows = tuple(
        blob.bbox for blob in connected_components(frame, colors=(7,))
    )
    borders = tuple(
        blob.bbox for blob in connected_components(frame, colors=(9, 11))
        if blob.area >= 13
    )
    return arrows, borders


def compact(frame):
    blobs = connected_components(frame, colors=(8, 9, 12, 14))
    return tuple(
        sorted((blob.color, blob.top_left) for blob in blobs
               if blob.size in ((2, 2), (4, 4)))
    )


def key_events(root, max_depth, max_states):
    root_node = root.clone()
    root_key = frame_key(root_node)
    best_paths = {root_key: ()}
    events = {}
    inverse = {1: 2, 2: 1, 3: 4, 4: 3}
    queue = deque(((root_node, ()),))
    inverse_resets = 0

    while queue and len(best_paths) < max_states:
        node, path = queue.popleft()
        for move in generic_moves(node.frame()):
            event_key = (frame_key(node), move)
            if event_key not in events:
                events[event_key] = (path, node.clone(), move)
        if len(path) >= max_depth:
            continue
        backup = node.clone()
        for action in (1, 2, 3, 4):
            before = frame_key(node)
            safe_step(node, action)
            child_key = frame_key(node)
            if child_key == before:
                continue
            child_path = path + (action,)
            if child_key not in best_paths:
                best_paths[child_key] = child_path
                queue.append((node.clone(), child_path))
            safe_step(node, inverse[action])
            if frame_key(node) != before:
                inverse_resets += 1
                node = backup.clone()

    if inverse_resets:
        print("inverse_resets", inverse_resets, flush=True)
    return tuple(events.values()), len(best_paths)


def probe(env):
    with open("frontier_scaffold.json") as stream:
        prefix = json.load(stream)["staged_prefix_actions"]
    for action in prefix[:318]:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    entry = env.clone()
    base_level = int(entry.levels_completed)

    recorder = Recorder(entry.clone())
    solve_parallel_wrapped_bridge_carrier_peg_solitaire(recorder)
    print("known", len(recorder.path), int(recorder.levels_completed), flush=True)

    bridge_first_seed = (
        3, 3, 1, 1, 4, 4, 4,
        (6, 43, 13), (6, 43, 25),
        3, 3, 3, 2, 2, 4, 4, 4, 2,
        (6, 43, 43), (6, 43, 55),
        1, 3, 3, 3, 1, 1, 3, 3, 3,
        (6, 7, 13), (6, 7, 25),
    )
    bridge_first_phase_two = bridge_first_seed + (
        4, 4, 4, 2, 2, 3, 3, 2,
        (6, 13, 43), (6, 13, 55),
        (6, 13, 55), (6, 25, 55),
        (6, 25, 55), (6, 37, 55),
        (6, 37, 55), (6, 49, 55),
        (6, 43, 55), (6, 55, 55),
        (6, 5, 55), (6, 17, 55),
    )

    beam_size = int(os.environ.get("OPT_BEAM", "8"))
    max_depth = int(os.environ.get("OPT_EVENTS", "23"))
    key_depth = int(os.environ.get("OPT_KEY_DEPTH", "16"))
    key_states = int(os.environ.get("OPT_KEY_STATES", "220"))
    cost_bound = int(os.environ.get("OPT_COST", "144"))

    # cost, node, public action path, last physical move
    event_offset = 0
    if os.environ.get("OPT_SEED_BRIDGE_PHASE2") == "1":
        seed = entry.clone()
        for action in bridge_first_phase_two:
            safe_step(seed, action)
        frontier = [(len(bridge_first_phase_two), seed,
                     bridge_first_phase_two,
                     ("peg", (54, 4), (54, 16)))]
        best = {frame_key(seed): len(bridge_first_phase_two)}
        event_offset = 8
    elif os.environ.get("OPT_SEED_BRIDGE_FIRST") == "1":
        seed = entry.clone()
        for action in bridge_first_seed:
            safe_step(seed, action)
        frontier = [(len(bridge_first_seed), seed, bridge_first_seed,
                     ("peg", (12, 6), (24, 6)))]
        best = {frame_key(seed): len(bridge_first_seed)}
        event_offset = 3
    else:
        frontier = [(0, entry, (), None)]
        best = {frame_key(entry): 0}
    for event_depth in range(max_depth):
        children = {}
        closures = 0
        aligned_states = 0
        for cost, root, path, last in frontier:
            outcomes, closure_size = key_events(root, key_depth, key_states)
            closures += closure_size
            aligned_states += len(outcomes)
            for keys, aligned, move in outcomes:
                if (
                    last is not None
                    and move[1] == last[2]
                    and move[2] == last[1]
                ):
                    continue
                child_cost = cost + len(keys) + 2
                if child_cost >= cost_bound:
                    continue
                child = aligned.clone()
                before = frame_key(child)
                move_clicks = clicks(move)
                for action in move_clicks:
                    safe_step(child, action)
                after = frame_key(child)
                if after == before:
                    continue
                child_path = path + keys + move_clicks
                if int(child.levels_completed) > base_level:
                    print("solution", len(child_path), event_depth + 1,
                          child_path, flush=True)
                    return
                known = best.get(after)
                if known is not None and known <= child_cost:
                    continue
                best[after] = child_cost
                prior = children.get(after)
                candidate = (child_cost, child, child_path, move)
                if prior is None or child_cost < prior[0]:
                    children[after] = candidate
        ordered = sorted(
            children.values(),
            key=lambda item: (item[0], compact(item[1].frame()),
                              phase(item[1].frame())),
        )
        if (
            os.environ.get("OPT_BRIDGE_FIRST") == "1"
            and event_offset == 0
            and event_depth == 1
        ):
            ordered = [item for item in ordered if (
                (8, (54, 42)) in compact(item[1].frame())
                and (14, (12, 6)) in compact(item[1].frame())
            )]
        frontier = ordered[:beam_size]
        print("event", event_offset + event_depth + 1,
              "frontier", len(frontier),
              "generated", len(children), "closures", closures,
              "aligned", aligned_states,
              "costs", tuple(item[0] for item in frontier),
              "pieces", tuple(compact(item[1].frame()) for item in frontier[:3]),
              flush=True)
        if os.environ.get("OPT_VERBOSE") == "1":
            for candidate in frontier:
                print("candidate", event_offset + event_depth + 1,
                      candidate[0],
                      candidate[2], candidate[3],
                      compact(candidate[1].frame()), flush=True)
        if not frontier:
            return


arena.run_program("lf52", probe)
