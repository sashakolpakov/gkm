"""Switch-macro beam that preserves autonomous-mover direction and histories."""
import json
import sys
import time
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import MOVES, _avatar_pos


DIRS = {1: (-6, 0), 2: (6, 0), 3: (0, -6), 4: (0, 6)}


def observations(env):
    frame = np.asarray(env.frame())
    avatar = _avatar_pos(frame)
    hr, hc = np.where(frame == 14)
    helper = None if not len(hr) else (int(hr.min()), int(hc.min()))
    _, mc = np.where(frame[1:4] == 9)
    marker = None if not len(mc) else (1, int(mc.min()))
    visual = np.where(
        np.isin(frame[1:62], (8, 9, 11, 14, 15)),
        frame[1:62],
        0,
    ).tobytes()
    return avatar, helper, marker, visual


def movement_possible(frame, avatar, action):
    dr, dc = DIRS[action]
    r, c = avatar[0] + dr, avatar[1] + dc
    if r < 0 or c < 0 or r + 5 > 64 or c + 5 > 64:
        return False
    return bool(np.isin(
        frame[r:r + 5, c:c + 5], (5, 8, 9, 11)
    ).all())


def moved_child(node, action, parent_avatar, parent_helper):
    try:
        child = node.clone()
        child.step(action)
        avatar, helper, marker, visual = observations(child)
    except (IndexError, ValueError):
        return None
    if child.terminal() or helper is None or avatar == parent_avatar:
        return None
    delta = (
        helper[0] - parent_helper[0],
        helper[1] - parent_helper[1],
    )
    return child, avatar, helper, marker, visual, delta


def post_use_heading(child, fallback):
    avatar, helper, _, _ = observations(child)
    frame = np.asarray(child.frame())
    for action in MOVES:
        if not movement_possible(frame, avatar, action):
            continue
        outcome = moved_child(child, action, avatar, helper)
        if outcome is not None:
            delta = outcome[-1]
            return fallback if delta == (0, 0) else delta
    return fallback


def movement_commits(
    root, heading, base, max_depth=22, max_states=350, histories=3,
):
    avatar, helper, _, visual = observations(root)
    queue = deque([
        (root.clone(), [], heading, avatar, helper, helper[0])
    ])
    seen = {(visual, heading, 0)}
    groups = defaultdict(list)
    best_row = helper[0]
    expanded = 0

    while queue and expanded < max_states:
        (
            node, path, direction, avatar, helper, path_best,
        ) = queue.popleft()
        expanded += 1
        if len(path) >= max_depth:
            continue

        try:
            child = node.clone()
            before_marker = observations(node)[2]
            child.step(5)
            child_obs = observations(child)
        except (IndexError, ValueError):
            child = None
        if (
            child is not None
            and not child.terminal()
            and child_obs[1] is not None
            and child_obs[2] != before_marker
        ):
            macro = path + [5]
            if int(child.levels_completed) > base:
                return macro, [], expanded, path_best
            child_direction = direction
            key = (child_obs[3], child_direction)
            group = groups[key]
            if len(group) < histories:
                group.append(
                    (
                        child, macro, child_direction, path_best,
                        avatar[0] in (26, 32), avatar,
                    )
                )

        frame = np.asarray(node.frame())
        for action in MOVES:
            if not movement_possible(frame, avatar, action):
                continue
            outcome = moved_child(node, action, avatar, helper)
            if outcome is None:
                continue
            child, child_avatar, child_helper, _, visual, delta = outcome
            child_path = path + [action]
            if int(child.levels_completed) > base:
                return child_path, [], expanded, min(best_row, child_helper[0])
            child_direction = direction if delta == (0, 0) else delta
            key = (visual, child_direction, len(child_path))
            if key in seen:
                continue
            seen.add(key)
            best_row = min(best_row, child_helper[0])
            child_best = min(path_best, child_helper[0])
            queue.append(
                (
                    child, child_path, child_direction,
                    child_avatar, child_helper, child_best,
                )
            )
    children = [item for group in groups.values() for item in group]
    return None, children, expanded, best_row


def barrier_area(env):
    return int(np.count_nonzero(np.asarray(env.frame()) == 15))


def search(root, max_stages=14, beam=6, histories=3):
    base = int(root.levels_completed)
    frontier = [
        (root.clone(), [], (0, -6), 56, 0, None, frozenset())
    ]
    started = time.time()
    for stage in range(max_stages):
        groups = defaultdict(list)
        stats = []
        for (
            node, prefix, heading, branch_best, reset_uses, last_use,
            used_rows,
        ) in frontier:
            win, children, expanded, local_best = movement_commits(
                node, heading, base, histories=histories
            )
            stats.append((expanded, len(children), local_best))
            if win is not None:
                return prefix + win
            for (
                child, macro, child_heading, macro_best, is_reset,
                use_pos,
            ) in children:
                child_reset_uses = reset_uses + int(is_reset)
                if child_reset_uses > 1:
                    continue
                if use_pos == last_use:
                    continue
                combined = prefix + macro
                obs = observations(child)
                key = (obs[3], child_heading)
                group = groups[key]
                if len(group) < histories:
                    group.append(
                        (
                            child, combined, child_heading,
                            min(branch_best, macro_best), child_reset_uses,
                            use_pos,
                            used_rows | {use_pos[0]},
                        )
                    )
        ranked = []
        for group in groups.values():
            for (
                child, path, heading, branch_best, reset_uses, last_use,
                used_rows,
            ) in group:
                helper = observations(child)[1]
                rank = (
                    (
                        branch_best, -len(used_rows), helper[0], reset_uses,
                        barrier_area(child), len(path),
                    )
                )
                ranked.append(
                    (
                        rank, child, path, heading,
                        branch_best, reset_uses, last_use, used_rows,
                    )
                )
        ranked.sort(key=lambda item: item[0])
        frontier = [
            (
                child, path, heading, branch_best, reset_uses,
                last_use, used_rows,
            )
            for (
                _, child, path, heading, branch_best, reset_uses,
                last_use, used_rows
            )
            in ranked[:beam]
        ]
        print(
            "stage", stage + 1,
            "groups", len(groups), "frontier", len(frontier),
            "stats", stats[:8],
            "best", None if not ranked else ranked[0][0],
            "path", None if not ranked else ranked[0][2],
            "seconds", round(time.time() - started, 1),
            flush=True,
        )
        if not frontier:
            break
    return None


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint:
        env.step(action)
    plan = search(env)
    print("plan", plan, flush=True)
    if plan:
        for action in plan:
            if env.terminal():
                break
            env.step(action)
    print("end", int(env.levels_completed), flush=True)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
