"""Bounded clone search from the first verified level-9 relay transition."""

import json
import os
import sys
from collections import deque
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import action_deltas, arr, color_counts, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)),
    ((48, 24), (36, 24)),
    ((42, 24), (30, 24)),
    ((36, 24), (24, 24)),
    ((30, 24), (18, 24)),
    ((18, 24), (18, 36)),
    ((18, 36), (30, 36)),
    ((24, 36), (36, 36)),
    ((30, 36), (42, 36)),
    ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
    ((48, 30), (36, 30)),
    ((48, 36), (36, 36)),
    ((36, 30), (36, 42)),
)


def play_move(env, move):
    source, destination = move
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def board(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    fixed_bridges = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    return (
        frozenset(holes | pegs | bridges | carriers | fixed_bridges),
        frozenset(pegs),
        frozenset(bridges),
        frozenset(carriers),
        frozenset(fixed_bridges),
    )


def summary(frame):
    cells, pegs, bridges, carriers, fixed_bridges = board(frame)
    borders = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11))
        if blob.area >= 13
    )
    return {
        "colors": color_counts(frame),
        "cells": tuple(sorted(cells)),
        "pegs": tuple(sorted(pegs)),
        "bridges": tuple(sorted(bridges)),
        "carriers": tuple(sorted(carriers)),
        "fixed_bridges": tuple(sorted(fixed_bridges)),
        "borders": borders,
        "special": tuple(
            (blob.color, blob.bbox, blob.size, blob.area)
            for blob in connected_components(frame, colors=(7, 9, 11, 12, 14, 15))
        ),
    }


def click_macros(frame):
    cells, pegs, bridges, _, fixed_bridges = board(frame)
    occupied = pegs | bridges | fixed_bridges
    macros = []
    for source in sorted(pegs | bridges):
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                midpoint in occupied
                and destination in cells
                and destination not in occupied
            ):
                macros.append((
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ))
    return tuple(macros)


def compact(frame):
    cells, pegs, bridges, carriers, fixed_bridges = board(frame)
    return (
        tuple(sorted(pegs)), tuple(sorted(bridges)),
        tuple(sorted(carriers)), tuple(sorted(fixed_bridges)),
        (min(c[0] for c in cells), min(c[1] for c in cells),
         max(c[0] for c in cells), max(c[1] for c in cells)),
    )


def visible_key_graph(root, max_depth=20, max_states=300):
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    observations = []
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (action,)
            observations.append((child_path, compact(child.frame())))
            queue.append((child, child_path))
    return len(seen), observations


def unlocked_key_contexts(root, max_depth=10, max_states=1000):
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    unlocked = []
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        for action in (1, 2, 3):
            child = node.clone()
            safe_step(child, action)
            if key(child) != key(node):
                unlocked.append((path, action, compact(node.frame()),
                                 compact(child.frame())))
        if len(path) >= max_depth:
            continue
        for macro in click_macros(node.frame()):
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (macro,)
            queue.append((child, child_path))
    return len(seen), unlocked


def actual_click_moves(root):
    cells, pegs, bridges, _, fixed_bridges = board(root.frame())
    before = compact(root.frame())
    found = []
    for source in sorted(pegs | bridges | fixed_bridges):
        for destination in sorted(cells):
            if source == destination:
                continue
            child = root.clone()
            safe_step(child, (6, source[1] + 1, source[0] + 1))
            safe_step(child, (6, destination[1] + 1, destination[0] + 1))
            after = compact(child.frame())
            if after != before or child.levels_completed > root.levels_completed:
                found.append((source, destination, after,
                              int(child.levels_completed)))
    return tuple(found)


def click_graph(root, max_depth=12, max_states=200):
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    observations = [((), compact(root.frame()))]
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for macro in click_macros(node.frame()):
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (macro,)
            observations.append((child_path, compact(child.frame())))
            queue.append((child, child_path))
    return tuple(observations)


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def search(root, max_cost=28, max_states=5000):
    base_level = int(root.levels_completed)
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {key(root): 0}
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)) or cost >= max_cost:
            continue
        macros = tuple((action,) for action in (3, 4))
        macros += click_macros(node.frame())
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > max_cost:
                continue
            child = node.clone()
            for action in macro:
                safe_step(child, action)
                if child.levels_completed > base_level:
                    return path + (macro,), len(best), child_cost
            child_key = key(child)
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, path + (macro,)))
    return None, len(best), None


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for move in FIRST_RELAY:
        play_move(env, move)

    print("stage_entry", int(env.levels_completed), summary(env.frame()))
    deltas = action_deltas(env)
    print("stage_keys", {
        action: (delta["count"] - (1 if delta["bbox"] is not None and delta["bbox"][0] == 0 else 0), delta["bbox"])
        for action, delta in deltas.items()
    })
    print("stage_click_macros", click_macros(env.frame()))
    special_actions = (
        (6, 59, 25), (6, 60, 25), (6, 59, 24), (6, 61, 26),
    )
    special_deltas = action_deltas(env, special_actions)
    print("stage_special_clicks", {
        action: (delta["count"], delta["bbox"], delta["samples"][:16])
        for action, delta in special_deltas.items()
    })
    long_stage = env.clone()
    stage_rewards = []
    for count in range(1, 81):
        safe_step(long_stage, 4)
        if long_stage.levels_completed > env.levels_completed:
            stage_rewards.append((count, int(long_stage.levels_completed)))
            break
    print("stage_long_right", stage_rewards,
          int(long_stage.levels_completed), compact(long_stage.frame()))
    for action in special_actions:
        clone = env.clone()
        safe_step(clone, action)
        once = summary(clone.frame())
        safe_step(clone, action)
        print("stage_special_repeat", action, once, summary(clone.frame()))
    if os.environ.get("L9_SPECIAL_ONLY") == "1":
        return
    contextual = []
    for macro in click_macros(env.frame()):
        node = env.clone()
        for action in macro:
            safe_step(node, action)
        children = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            children.append((action, compact(child.frame())))
        contextual.append((macro, compact(node.frame()), children))
    print("stage_macro_keys", contextual)
    unload = click_macros(env.frame())[1]
    unloaded = env.clone()
    for action in unload:
        safe_step(unloaded, action)
    unloaded_size, unloaded_graph = visible_key_graph(
        unloaded, max_depth=24, max_states=400
    )
    print("stage_unloaded_key_graph", unloaded_size, unloaded_graph)
    hidden_traces = []
    for lead in (6, 7, 8):
        for direction in (1, 2, 3, 4):
            node = unloaded.clone()
            for _ in range(lead):
                safe_step(node, 4)
            baseline = key(node)
            changes = []
            prior = baseline
            for count in range(1, 13):
                safe_step(node, direction)
                current = key(node)
                if current != prior:
                    changes.append((count, compact(node.frame())))
                prior = current
            if changes:
                hidden_traces.append((lead, direction, changes))
    print("stage_hidden_traces", hidden_traces)
    if os.environ.get("L9_CLICK_GRAPH") == "1":
        print("stage_click_graph", click_graph(env))
    if os.environ.get("L9_FRONTIER") == "1":
        frontier = env.clone()
        for action in (
            (6, 23, 37), (6, 11, 37),
            (6, 11, 37), (6, 23, 37),
        ):
            safe_step(frontier, action)
        print("frontier_two", int(frontier.levels_completed),
              summary(frontier.frame()))
        frontier_deltas = action_deltas(frontier)
        print("frontier_two_keys", {
            action: (delta["count"], delta["bbox"])
            for action, delta in frontier_deltas.items()
        })
        size, graph = visible_key_graph(frontier, max_depth=20, max_states=200)
        print("frontier_two_graph", size, graph)
        print("frontier_two_clicks", actual_click_moves(frontier))
        long_frontier = frontier.clone()
        frontier_rewards = []
        for count in range(1, 81):
            safe_step(long_frontier, 4)
            if long_frontier.levels_completed > frontier.levels_completed:
                frontier_rewards.append((count, int(long_frontier.levels_completed)))
                break
        print("frontier_two_long_right", frontier_rewards,
              int(long_frontier.levels_completed), compact(long_frontier.frame()))
        bridge_frontier = env.clone()
        for action in (
            (6, 23, 37), (6, 11, 37),
            4,
            (6, 11, 37), (6, 23, 37),
            (6, 17, 37), (6, 29, 37),
        ):
            safe_step(bridge_frontier, action)
        print("frontier_bridge", int(bridge_frontier.levels_completed),
              summary(bridge_frontier.frame()))
        bridge_deltas = action_deltas(bridge_frontier)
        print("frontier_bridge_keys", {
            action: (delta["count"], delta["bbox"])
            for action, delta in bridge_deltas.items()
        })
        print("frontier_bridge_clicks", actual_click_moves(bridge_frontier))
    graph_size, graph = visible_key_graph(env)
    print("stage_key_graph", graph_size, graph)
    shifted = env.clone()
    for offset in range(15):
        print("stage_shift_macros", offset, compact(shifted.frame()),
              click_macros(shifted.frame()))
        safe_step(shifted, 4)
    click_states, unlocked = unlocked_key_contexts(env)
    print("stage_unlocked", click_states, unlocked)
    shifted = env.clone()
    shifted_contexts = []
    for offset in range(15):
        click_states, unlocked = unlocked_key_contexts(
            shifted, max_depth=8, max_states=500
        )
        if unlocked:
            shifted_contexts.append((offset, click_states, unlocked))
        safe_step(shifted, 4)
    print("stage_shifted_unlocked", shifted_contexts)
    if os.environ.get("L9_ENUM") == "1":
        shifted = env.clone()
        for offset in range(9):
            print("stage_actual_moves", offset, actual_click_moves(shifted))
            safe_step(shifted, 4)
    sequences = (
        (4,), (7,), (4, 7), (7, 4),
        (4, 4), (4, 4, 4), (4, 4, 4, 4),
        (7, 7),
    )
    for actions in sequences:
        clone = env.clone()
        for action in actions:
            safe_step(clone, action)
        print("stage_sequence", actions, int(clone.levels_completed),
              summary(clone.frame()))

    if os.environ.get("L9_SEARCH") == "1":
        solution, explored, cost = search(
            env,
            max_cost=int(os.environ.get("L9_MAX_COST", "28")),
            max_states=int(os.environ.get("L9_MAX_STATES", "5000")),
        )
        print("stage_search", explored, cost, solution)
        if solution is not None:
            clone = env.clone()
            for macro in solution:
                for action in macro:
                    safe_step(clone, action)
            print("stage_result", int(clone.levels_completed), summary(clone.frame()))


arena.run_program("lf52", probe)
