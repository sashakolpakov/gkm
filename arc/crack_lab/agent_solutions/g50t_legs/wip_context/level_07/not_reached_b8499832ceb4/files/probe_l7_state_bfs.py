"""Bounded level-7 search over visible state plus helper heading."""
import json
import heapq
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import MOVES, _avatar_pos
DIRS = {1: (-6, 0), 2: (6, 0), 3: (0, -6), 4: (0, 6)}
PREFIX = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]


def observations(env):
    frame = np.asarray(env.frame())
    avatar = _avatar_pos(frame)
    hr, hc = np.where(frame == 14)
    helper = None if not len(hr) else (int(hr.min()), int(hc.min()))
    _, mc = np.where(frame[1:4] == 9)
    marker = None if not len(mc) else (1, int(mc.min()))
    world = np.where(
        np.isin(frame[1:62], (8, 9, 11, 14, 15)),
        frame[1:62],
        0,
    )
    return avatar, helper, marker, world.tobytes()


def movement_possible(frame, avatar, action):
    dr, dc = DIRS[action]
    r, c = avatar[0] + dr, avatar[1] + dc
    if r < 0 or c < 0 or r + 5 > 64 or c + 5 > 64:
        return False
    target = frame[r:r + 5, c:c + 5]
    return bool(np.isin(target, (5, 8, 9, 11)).all())


def search(
    root, max_states=12000, max_depth=100, histories=1,
    initial_heading=(-6, 0),
):
    base = int(root.levels_completed)
    avatar, helper, marker, visual = observations(root)
    counter = 0
    queue = [
        (
            (helper[0], 0, 0, 0), counter,
            root.clone(), [], initial_heading, helper, 0, 0, helper[0],
        )
    ]
    seen = {(visual, initial_heading, 0, 0): 1}
    started = time.time()
    expanded = 0
    best_row = helper[0]
    valid_use_positions = set()

    while queue and expanded < max_states:
        (
            _, _, node, path, heading, old_helper,
            move_count, use_count, branch_best,
        ) = heapq.heappop(queue)
        expanded += 1
        if expanded % 250 == 0:
            print(
                "progress", expanded, "queued", len(queue),
                "seen", len(seen), "depth", len(path),
                "best_row", best_row,
                "seconds", round(time.time() - started, 1),
                flush=True,
            )
        if len(path) >= max_depth or node.terminal():
            continue

        parent_frame = np.asarray(node.frame())
        parent_avatar, parent_helper, parent_marker, _ = observations(node)
        actions = list(MOVES) if move_count < 90 else []
        if use_count < 12:
            actions.append(5)
        for action in actions:
            try:
                child = node.clone()
                child.step(action)
                if int(child.levels_completed) > base:
                    return path + [action], expanded
                if child.terminal():
                    continue
                avatar, helper, marker, visual = observations(child)
            except (IndexError, ValueError):
                continue

            if helper is None:
                continue
            if action == 5:
                if marker == parent_marker:
                    continue
                if parent_avatar not in valid_use_positions:
                    valid_use_positions.add(parent_avatar)
                    print(
                        "valid_use", parent_avatar,
                        "path_len", len(path), flush=True,
                    )
                child_heading = heading
                child_moves = move_count
                child_uses = use_count + 1
            else:
                if avatar == parent_avatar:
                    continue
                delta = (
                    helper[0] - parent_helper[0],
                    helper[1] - parent_helper[1],
                )
                child_heading = heading if delta == (0, 0) else delta
                child_moves = move_count + 1
                child_uses = use_count

            best_row = min(best_row, helper[0])
            child_best = min(branch_best, helper[0])
            key = (visual, child_heading, child_moves, child_uses)
            count = seen.get(key, 0)
            if count >= histories:
                continue
            seen[key] = count + 1
            counter += 1
            child_path = path + [action]
            heapq.heappush(
                queue,
                (
                    (
                        child_best, -child_uses,
                        len(child_path), child_moves,
                    ),
                    counter, child, child_path, child_heading, helper,
                    child_moves, child_uses, child_best,
                ),
            )
    return None, expanded


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint:
        env.step(action)
    node = env.clone()
    for action in PREFIX:
        node.step(action)
    tail, expanded = search(
        node, max_states=30000, max_depth=120,
        histories=1, initial_heading=(0, -6),
    )
    plan = None if tail is None else PREFIX + tail
    print("plan", plan, "expanded", expanded, flush=True)
    if plan:
        for action in plan:
            env.step(action)
            if env.terminal():
                break
    print("end", int(env.levels_completed), flush=True)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
