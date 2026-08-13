"""In-place branch-and-bound search using verified action-7 restoration."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _movable_bridge_board
from perception import arr, connected_components, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def peg_count(frame):
    return sum(
        blob.color == 14 and blob.size == (4, 4)
        for blob in connected_components(frame, colors=(14,))
    )


def color8_moves(frame):
    blobs = connected_components(frame, colors=(1, 8, 12, 14, 15))
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
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    destinations = slots | carriers
    occupied = pegs | movable
    out = []
    for kind, sources in (("peg", pegs), ("movable", movable)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr,
                               source[1] + 2 * dc)
                if destination not in destinations or destination in occupied:
                    continue
                legal = (
                    midpoint in pegs
                    if kind == "movable"
                    else midpoint in pegs | movable | fixed
                )
                if legal:
                    capture = kind == "peg" and midpoint in pegs
                    out.append((0 if capture else 2, source, destination))
    return out


def color9_moves(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    occupied = bridges | pegs
    out = []
    for kind, sources in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr,
                               source[1] + 2 * dc)
                if (
                    midpoint in occupied
                    and destination in slots | carriers
                    and destination not in occupied
                ):
                    capture = kind == "peg" and midpoint in pegs
                    out.append((0 if capture else 2, source, destination))
    return out


def lattice_moves(frame):
    candidates = list(color8_moves(frame)) + list(color9_moves(frame))
    candidates.extend(
        (0 if kind == "capture" else 1, source, destination)
        for kind, source, destination in _bridge_carrier_moves(frame)
    )
    unique = {}
    for rank, source, destination in candidates:
        key = (source, destination)
        unique[key] = min(rank, unique.get(key, rank))
    return tuple(
        (source, destination)
        for (source, destination), _ in sorted(
            unique.items(), key=lambda item: (item[1], item[0])
        )
    )


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "4"))
    bound = int(os.environ.get("OPT_COST", "50"))
    max_expanded = int(os.environ.get("OPT_STATES", "20000"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    root = env.clone() if desired == 1 else None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            root = env.clone()
            break
        prior = current

    base_level = int(root.levels_completed)
    print("undo_entry", desired, base_level, peg_count(root.frame()),
          flush=True)
    best = {state_key(root): 0}
    path = []
    solution = None
    expanded = 0

    def visit(node, cost, last_key=None):
        nonlocal expanded, solution
        if solution is not None or expanded >= max_expanded:
            return
        remaining = 2 * max(0, peg_count(node.frame()) - 1)
        if cost + remaining > bound:
            return
        expanded += 1
        if expanded % 100 == 0:
            print("undo_progress", expanded, len(best), cost, len(path),
                  peg_count(node.frame()), flush=True)

        before_key = state_key(node)
        before_frame = arr(node.frame()).copy()
        for source, destination in lattice_moves(node.frame()):
            if cost + 2 > bound:
                continue
            clicks = (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            child = node.clone()
            for action in clicks:
                safe_step(child, action)
            path.extend(clicks)
            if int(child.levels_completed) > base_level:
                solution = tuple(path)
                return
            child_key = state_key(child)
            child_cost = cost + 2
            if child_key != before_key and child_cost < best.get(
                    child_key, 10 ** 9):
                best[child_key] = child_cost
                visit(child, child_cost)
            del path[-2:]
            if solution is not None or expanded >= max_expanded:
                return

        inverse = {1: 2, 2: 1, 3: 4, 4: 3}
        for action in (4, 3, 2, 1):
            if cost + 1 > bound or (
                    last_key is not None and action == inverse[last_key]):
                continue
            child = node.clone()
            safe_step(child, action)
            child_key = state_key(child)
            if child_key == before_key:
                continue
            path.append(action)
            child_cost = cost + 1
            if child_cost < best.get(child_key, 10 ** 9):
                best[child_key] = child_cost
                visit(child, child_cost, action)
            path.pop()
            if solution is not None or expanded >= max_expanded:
                return

    try:
        visit(root, 0)
    except Exception as error:
        print("undo_error", type(error).__name__, repr(error), flush=True)
        raise
    print("undo_result", desired, bound, expanded, len(best),
          None if solution is None else len(solution), solution, flush=True)


arena.run_program("lf52", probe)
