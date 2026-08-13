"""Test every reachable first-board load layout for a rail turn."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def parse(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    cells = frozenset(
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 9, 12, 14)
    )
    pegs = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    )
    bridges = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    )
    carrier = next(
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    )
    return cells, pegs, bridges, carrier


def load_paths(frame, goal_kind="P"):
    cells, pegs, bridges, carrier = parse(frame)
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    goals = {}
    while queue:
        (state_pegs, state_bridges), path = queue.popleft()
        occupied = state_pegs | state_bridges
        for kind, source in (
            tuple(("P", cell) for cell in sorted(state_pegs))
            + tuple(("B", cell) for cell in sorted(state_bridges))
        ):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in cells
                    or destination in occupied
                ):
                    continue
                child_pegs = set(state_pegs)
                child_bridges = set(state_bridges)
                if kind == "P":
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source)
                    child_bridges.add(destination)
                child = (frozenset(child_pegs), frozenset(child_bridges))
                clicks = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                child_path = path + clicks
                if destination == carrier and kind == goal_kind:
                    goals.setdefault(child, child_path)
                    continue
                if child not in seen:
                    seen.add(child)
                    queue.append((child, child_path))
    return tuple(sorted(goals.items(), key=lambda item: len(item[1]))), len(seen)


def puzzle_key(env):
    return arr(env.frame())[1:, :].tobytes()


def stage_pieces(frame):
    return tuple(
        sorted(
            (blob.color, blob.top_left, blob.size, blob.area)
            for blob in connected_components(frame, colors=(9, 11, 12, 14, 15))
            if blob.color not in (9, 11) or blob.area >= 12
        )
    )


def stage_bridges(frame):
    return tuple(sorted(
        blob.top_left
        for blob in connected_components(frame, colors=(9,))
        if blob.size == (4, 4) and blob.area == 12
    ))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    goal_kind = os.environ.get("OPT_LOAD_KIND", "P")
    goals, states = load_paths(env.frame(), goal_kind=goal_kind)
    print("models", states, len(goals),
          tuple(sorted({len(path) for _, path in goals})), flush=True)
    if os.environ.get("OPT_ONLY_MODELS") == "1":
        details = tuple(
            (len(path) // 2, len(state[0]), tuple(sorted(state[0])),
             tuple(sorted(state[1])), path)
            for state, path in goals
        )
        if os.environ.get("OPT_SUMMARY") == "1":
            print("load_summary", goal_kind, len(details), details[:10],
                  flush=True)
        else:
            print("load_goals", details, flush=True)
        return
    changed = []
    layouts = {}
    for index, (_, path) in enumerate(goals):
        root = env.clone()
        for action in path:
            safe_step(root, action)
        layout = stage_bridges(root.frame())
        prior = layouts.get(layout)
        if prior is None or len(path) < prior[0]:
            layouts[layout] = (len(path), path)
        for offset in range(9):
            for action in (1, 2):
                before = puzzle_key(root)
                base_level = int(root.levels_completed)
                safe_step(root, action)
                if puzzle_key(root) != before or root.levels_completed > base_level:
                    changed.append((index, len(path), offset, action,
                                    int(root.levels_completed),
                                    stage_pieces(root.frame()), path))
                    break
            else:
                safe_step(root, 4)
                continue
            break
        if (index + 1) % 10 == 0:
            print("progress", index + 1, len(changed), flush=True)
    print("layouts", len(layouts),
          tuple((cost, layout) for layout, (cost, _) in sorted(
              layouts.items(), key=lambda item: item[1][0]
          )), flush=True)
    print("vertical_changes", changed, flush=True)


arena.run_program("lf52", probe)
