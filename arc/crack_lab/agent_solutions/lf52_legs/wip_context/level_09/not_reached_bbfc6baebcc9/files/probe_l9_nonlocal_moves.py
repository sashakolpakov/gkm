"""Test whether level 9 accepts lattice transfers longer than one jump."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def board(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    return holes, bridges, pegs, fixed, carriers


def nonlocal_results(root):
    holes, bridges, pegs, fixed, carriers = board(root.frame())
    cells = holes | bridges | pegs | fixed | carriers
    occupied = bridges | pegs | fixed
    accepted = []
    tested = 0
    for kind, sources in (("P", pegs), ("B", bridges)):
        for source in sorted(sources):
            for destination in sorted(cells - occupied):
                distance = abs(destination[0] - source[0]) + abs(
                    destination[1] - source[1]
                )
                if not (
                    (destination[0] == source[0]
                     or destination[1] == source[1])
                    and distance > 12
                    and distance % 6 == 0
                ):
                    continue
                tested += 1
                child = root.clone()
                move(child, source, destination)
                _, child_bridges, child_pegs, _, _ = board(child.frame())
                if child_pegs != pegs or child_bridges != bridges:
                    accepted.append((kind, source, destination,
                                     tuple(sorted(child_pegs)),
                                     tuple(sorted(child_bridges))))
    return tested, tuple(accepted)


def diagonal_results(root):
    holes, bridges, pegs, fixed, carriers = board(root.frame())
    cells = holes | bridges | pegs | fixed | carriers
    occupied = bridges | pegs | fixed
    accepted = []
    candidates = []
    for kind, sources in (("P", pegs), ("B", bridges)):
        for source in sorted(sources):
            for dr, dc in ((-6, -6), (-6, 6), (6, -6), (6, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr,
                               source[1] + 2 * dc)
                if midpoint in occupied and destination in cells - occupied:
                    candidates.append((kind, source, midpoint, destination))
    for kind, source, midpoint, destination in candidates:
        child = root.clone()
        move(child, source, destination)
        _, child_bridges, child_pegs, _, _ = board(child.frame())
        if child_pegs != pegs or child_bridges != bridges:
            accepted.append((kind, source, midpoint, destination,
                             tuple(sorted(child_pegs)),
                             tuple(sorted(child_bridges))))
    return tuple(candidates), tuple(accepted)


def all_pair_results(root):
    holes, bridges, pegs, fixed, carriers = board(root.frame())
    cells = holes | bridges | pegs | fixed | carriers
    occupied = bridges | pegs | fixed
    accepted = []
    tested = 0
    base_level = int(root.levels_completed)
    for kind, sources in (("P", pegs), ("B", bridges)):
        for source in sorted(sources):
            for destination in sorted(cells - {source}):
                tested += 1
                child = root.clone()
                move(child, source, destination)
                _, child_bridges, child_pegs, _, _ = board(child.frame())
                if (child_pegs != pegs or child_bridges != bridges
                        or int(child.levels_completed) > base_level):
                    accepted.append((
                        kind, source, destination,
                        destination in occupied,
                        (destination[0] - source[0],
                         destination[1] - source[1]),
                        tuple(sorted(child_pegs)),
                        tuple(sorted(child_bridges)),
                        int(child.levels_completed),
                    ))
    return tested, tuple(accepted)


def bridge_attempt(root, source, destination):
    before = board(root.frame())[1]
    child = root.clone()
    move(child, source, destination)
    after = board(child.frame())[1]
    return before != after, tuple(sorted(after))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)

    all_context = os.environ.get("OPT_ALL_PAIRS")
    if all_context == "entry":
        print("all_pairs_entry", all_pair_results(env), flush=True)
        return

    print("nonlocal_entry", nonlocal_results(env), flush=True)
    print("diagonal_entry", diagonal_results(env), flush=True)

    for source, destination in FIRST_RELAY:
        move(env, source, destination)
    print("empty_midpoint_near", (
        ((18, 10), (18, 22), bridge_attempt(env, (18, 10), (18, 22))),
        ((24, 4), (24, 16), bridge_attempt(env, (24, 4), (24, 16))),
    ), flush=True)
    for _ in range(9):
        safe_step(env, 4)
    if all_context == "far":
        print("all_pairs_far", all_pair_results(env), flush=True)
        return
    print("empty_midpoint_far", (
        ((18, 52), (18, 40), bridge_attempt(env, (18, 52), (18, 40))),
        ((18, 58), (18, 46), bridge_attempt(env, (18, 58), (18, 46))),
    ), flush=True)
    print("nonlocal_far", nonlocal_results(env), flush=True)
    print("diagonal_far", diagonal_results(env), flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
