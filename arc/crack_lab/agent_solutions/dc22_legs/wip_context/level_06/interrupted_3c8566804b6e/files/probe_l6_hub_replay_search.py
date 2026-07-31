"""Path-only continuation of the verified level-6 hub component search."""
import heapq
import itertools
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import probe_l6_component_planner as planner
import solve


LEVEL_BUDGET = 295
GLOBAL_SEARCH = sys.argv[1:] == ["global"]
HUB_PREFIX = (
    []
    if GLOBAL_SEARCH
    else planner.PHYSICAL_ENTRY + planner.PHYSICAL_TO_HUB
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def reconstruct(base, path):
    node = base.clone()
    apply(node, path)
    return node


def abstract_key(env):
    """Retain irreversible floor changes while quotienting avatar location."""
    frame = perception.arr(env.frame())[:63].copy()
    position = planner.avatar(env)
    if position is not None:
        row, col = position
        frame[row:row + 2, col:col + 2] = 2
    component = tuple(sorted(
        position
        for position in planner.walk_closure(env)
        if position is not None
    ))
    return frame.tobytes(), component


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    apply(env, HUB_PREFIX)
    planner.FULL_AVATAR_CONTEXTS = True

    serial = itertools.count()
    heap = [(0, next(serial), tuple())]
    queued_frames = {
        abstract_key(env) if GLOBAL_SEARCH else planner.visible_key(env)
    }
    seen_components = set()
    seen_region_sets = set()
    best = (10**9, -1, -1)

    while heap and len(seen_components) < 5000:
        _, _, root_path_tuple = heapq.heappop(heap)
        root_path = list(root_path_tuple)
        root = reconstruct(env, root_path)
        by_position = planner.walk_closure(root)
        positions = sorted(
            position for position in by_position if position is not None
        )
        if not positions:
            continue

        canonical_position = positions[0]
        canonical, winning_walk = planner.replay_walk(
            root, by_position[canonical_position], base_level
        )
        if winning_walk is not None:
            winner = HUB_PREFIX + root_path + winning_walk
            print("REPLAY_WIN", len(winner), winner, flush=True)
            return
        component_key = (
            abstract_key(root)
            if GLOBAL_SEARCH
            else planner.visible_key(canonical)
        )
        if component_key in seen_components:
            continue
        seen_components.add(component_key)

        regions = tuple(sorted({
            planner.interaction_region(position)[0]
            for position in positions
        }))
        if regions not in seen_region_sets:
            seen_region_sets.add(regions)
            print(
                "REPLAY_REGIONS", regions, "positions", len(positions),
                "suffix", len(root_path), "path", root_path, flush=True,
            )

        exits = sorted(planner.solid_exits(root))
        metric = (
            min(position[0] for position in positions),
            max(position[1] for position in positions),
            len(positions),
        )
        if exits:
            print(
                "REPLAY_EXIT", exits, "metric", metric,
                "suffix", len(root_path), "path", root_path, flush=True,
            )
            for r0, c0, _, _ in exits:
                goal_path = by_position.get((r0, c0))
                if goal_path is None:
                    continue
                _, winning_walk = planner.replay_walk(
                    root, goal_path, base_level
                )
                if winning_walk is not None:
                    winner = HUB_PREFIX + root_path + winning_walk
                    print("REPLAY_WIN", len(winner), winner, flush=True)
                    return

        improved = (
            metric[0] < best[0]
            or metric[1] > best[1]
            or metric[2] > best[2]
        )
        if improved:
            best = (
                min(best[0], metric[0]),
                max(best[1], metric[1]),
                max(best[2], metric[2]),
            )
            print(
                "REPLAY_PROGRESS", len(seen_components), len(heap),
                "metric", metric, "best", best,
                "suffix", len(root_path), "path", root_path, flush=True,
            )

        if GLOBAL_SEARCH:
            for walk_path in by_position.values():
                if not walk_path:
                    continue
                child_path = root_path + walk_path
                if len(child_path) > LEVEL_BUDGET:
                    continue
                walked = root.clone()
                apply(walked, walk_path)
                if walked.levels_completed > base_level:
                    print(
                        "REPLAY_WIN", len(child_path), child_path,
                        "movement", flush=True,
                    )
                    return
                walked_key = abstract_key(walked)
                if (
                    walked_key == component_key
                    or walked_key in queued_frames
                ):
                    continue
                queued_frames.add(walked_key)
                heapq.heappush(
                    heap,
                    (len(child_path), next(serial), tuple(child_path)),
                )

        local_outcomes = set()
        for label, target, walk_path, control in planner.control_sites(
                by_position):
            child_path = root_path + walk_path + [control]
            if len(HUB_PREFIX) + len(child_path) > LEVEL_BUDGET:
                continue
            walked = root.clone()
            apply(walked, walk_path)
            if planner.avatar(walked) != target:
                continue
            before_frame = perception.arr(walked.frame())[:63].copy()
            before = before_frame.tobytes()
            child = walked.clone()
            step(child, control)
            if child.levels_completed > base_level:
                winner = HUB_PREFIX + child_path
                print(
                    "REPLAY_WIN", len(winner), winner,
                    "control", label, flush=True,
                )
                return
            after = planner.visible_key(child)
            child_key = (
                abstract_key(child)
                if GLOBAL_SEARCH
                else planner.visible_key(child)
            )
            signature = (
                child_key
                if GLOBAL_SEARCH
                else planner.outcome_signature(before_frame, child)
            )
            if after == before or signature in local_outcomes:
                continue
            local_outcomes.add(signature)
            queued_key = child_key if GLOBAL_SEARCH else after
            if queued_key in queued_frames:
                continue
            queued_frames.add(queued_key)
            heapq.heappush(
                heap,
                (len(child_path), next(serial), tuple(child_path)),
            )

        if len(seen_components) % 50 == 0:
            print(
                "REPLAY_STATES", len(seen_components), len(heap),
                "suffix", len(root_path), "best", best, flush=True,
            )

    print(
        "REPLAY_DONE", len(seen_components), len(heap),
        "best", best, flush=True,
    )


arena.run_program("dc22", observe)
