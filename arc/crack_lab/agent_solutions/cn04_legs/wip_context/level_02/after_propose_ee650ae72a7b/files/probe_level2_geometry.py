"""Enumerate collision-free four-body assemblies by coincident handles."""
import sys
from itertools import combinations, product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


COLORS = {"A": 0, "B": 14, "C": 11, "D": 9}


def cells(frame, color):
    a = perception.arr(frame)
    return {
        (r, c)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == color
    }


def gray_cells(frame):
    return cells(frame, 8)


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def moved_handles(root, name):
    node = root.clone()
    if name != "A":
        select_color(node, COLORS[name])
    before = gray_cells(node.frame())
    node.step(1)
    return before - gray_cells(node.frame())


def normalize_body(body, handles):
    r0 = min(r for r, _ in body)
    c0 = min(c for _, c in body)
    return (
        frozenset((r - r0, c - c0) for r, c in body),
        frozenset((r - r0, c - c0) for r, c in handles),
    )


def rotate(body, handles):
    turned_body = {(c, -r) for r, c in body}
    turned_handles = {(c, -r) for r, c in handles}
    return normalize_body(turned_body, turned_handles)


def orientations(body, handles):
    out = []
    pose = normalize_body(body, handles)
    for _ in range(4):
        out.append(pose)
        pose = rotate(*pose)
    return out


def place(pose, shift):
    dr, dc = shift
    return tuple(
        frozenset((r + dr, c + dc) for r, c in part)
        for part in pose
    )


def compatible(first, second):
    body1, handles1 = first
    body2, handles2 = second
    return not (
        body1 & body2
        or body1 & handles2
        or handles1 & body2
    )


def collisions(poses):
    total = 0
    for index, first in enumerate(poses):
        for second in poses[index + 1:]:
            body1, handles1 = first
            body2, handles2 = second
            total += len(body1 & body2)
            total += len(body1 & handles2)
            total += len(handles1 & body2)
    return total


def placements_matching(poses, targets, wanted):
    found = []
    for turn, pose in enumerate(poses):
        for source, target in product(pose[1], targets):
            shift = (target[0] - source[0], target[1] - source[1])
            placed = place(pose, shift)
            if placed[1] & targets == wanted:
                item = (turn, shift, placed)
                if item not in found:
                    found.append(item)
    return found


def probe(env):
    play_level_1(env)
    root = env.clone()
    bodies = {name: cells(root.frame(), color) for name, color in COLORS.items()}
    handles = {name: moved_handles(root, name) for name in COLORS}
    poses = {
        name: orientations(bodies[name], handles[name])
        for name in COLORS
    }
    anchor = (frozenset(bodies["B"]), frozenset(handles["B"]))
    candidates = []
    for matched_d in map(frozenset, combinations(anchor[1], 2)):
        for d_turn, d_shift, d_pose in placements_matching(
            poses["D"], anchor[1], matched_d
        ):
            if d_pose[1] != matched_d:
                continue
            remaining_b = anchor[1] - d_pose[1]
            for c_turn, c_shift, c_pose in placements_matching(
                poses["C"], anchor[1], remaining_b
            ):
                if c_pose[1] & anchor[1] != remaining_b:
                    continue
                remaining_c = c_pose[1] - remaining_b
                for a_turn, a_shift, a_pose in placements_matching(
                    poses["A"], remaining_c, remaining_c
                ):
                    if a_pose[1] != remaining_c:
                        continue
                    candidates.append(
                        (
                            collisions((anchor, a_pose, c_pose, d_pose)),
                            (a_turn, a_shift),
                            (c_turn, c_shift),
                            (d_turn, d_shift),
                        )
                    )
    print("handles", {name: sorted(value) for name, value in handles.items()})
    print("candidates", len(candidates))
    for candidate in sorted(candidates)[:40]:
        print(candidate)


arena.run_program("cn04", probe)
