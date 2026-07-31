import importlib.util
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    action_deltas,
    bounded_bfs,
    bounded_replay_bfs,
    color_counts,
    connected_components,
)
from legs import (
    _selected_center,
    _selection_cycle,
    _shape_offsets,
    _survey_markers_and_swatches,
    paint_shapes_at_swatches_to_cover_ring_markers,
)


spec = importlib.util.spec_from_file_location("players", "players.py")
players = importlib.util.module_from_spec(spec)
spec.loader.exec_module(players)


def compact_components(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=2)
        if b.color != 5
    ]


def selected_shape(node):
    before = node.frame()
    try:
        center = _selected_center(node)
    except ValueError:
        return None, None, set()
    background = max(color_counts(before), key=color_counts(before).get)
    moved_frames = []
    votes = []
    for action in (UP, DOWN, LEFT, RIGHT):
        moved = node.clone()
        moved.step(action)
        after = moved.frame()
        moved_frames.append(after)
        for row in range(64):
            for col in range(64):
                if (
                    before[row][col] != after[row][col]
                    and after[row][col] == background
                    and before[row][col] not in (0, background, 4, 2)
                ):
                    votes.append(int(before[row][col]))
    if not votes:
        return center, None, set()
    color = max(set(votes), key=votes.count)
    cells = {
        (row, col)
        for after in moved_frames
        for row in range(64)
        for col in range(64)
        if before[row][col] == color and before[row][col] != after[row][col]
    }
    return center, color, cells


def selected_component(node):
    frame = node.frame()
    try:
        center = _selected_center(node)
    except ValueError:
        return None, set()
    candidates = [
        int(frame[center[0] + dr][center[1] + dc])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= center[0] + dr < 64 and 0 <= center[1] + dc < 64
        and int(frame[center[0] + dr][center[1] + dc]) not in (0, 2, 4, 5)
    ]
    if not candidates:
        return None, set()
    color = max(set(candidates), key=candidates.count)
    seeds = [
        (center[0] + dr, center[1] + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= center[0] + dr < 64 and 0 <= center[1] + dc < 64
        and int(frame[center[0] + dr][center[1] + dc]) == color
    ]
    cells = set(seeds)
    stack = list(seeds)
    while stack:
        row, col = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            point = row + dr, col + dc
            if (
                0 <= point[0] < 64
                and 0 <= point[1] < 64
                and point not in cells
                and int(frame[point[0]][point[1]]) == color
            ):
                cells.add(point)
                stack.append(point)
    return color, cells


def cross_signature(node):
    center = _selected_center(node)
    color, cells = selected_component(node)
    offsets = {(r - center[0], c - center[1]) for r, c in cells}
    return (
        center,
        color,
        len(offsets),
        min(r for r, _ in offsets),
        max(r for r, _ in offsets),
        min(c for _, c in offsets),
        max(c for _, c in offsets),
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    base = env.frame()
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(base))
    print("COMPONENTS", compact_components(base))
    print(
        "DELTAS",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        print(
            "AFTER",
            action,
            "reward",
            clone.levels_completed,
            "counts",
            color_counts(clone.frame()),
            "components",
            compact_components(clone.frame()),
        )
    centers = _selection_cycle(env)
    distance = 64 // 3 + 1
    markers, swatches = _survey_markers_and_swatches(
        env, len(centers), distance, 4, 2
    )
    print("CENTERS", centers)
    print("MARKERS", sorted(markers.items()))
    print("SWATCHES", {k: sorted(v) for k, v in sorted(swatches.items())})
    for index in range(len(centers)):
        center, color, offsets = _shape_offsets(
            env, index, len(centers), distance
        )
        print("SHAPE", index, center, color, sorted(offsets))
    trial = env.clone()
    try:
        paint_shapes_at_swatches_to_cover_ring_markers(trial)
        print("EXISTING_LEG", trial.levels_completed, "ok")
    except Exception as exc:
        print("EXISTING_LEG", trial.levels_completed, type(exc).__name__, str(exc))
    for name, path in (
        ("D8", [DOWN] * 8),
        ("D8U8", [DOWN] * 8 + [UP] * 8),
        ("R6U6", [RIGHT] * 6 + [UP] * 6),
        ("R6U8", [RIGHT] * 6 + [UP] * 8),
    ):
        node = env.clone()
        states = [cross_signature(node)]
        for action in path:
            node.step(action)
            try:
                states.append(cross_signature(node))
            except ValueError:
                states.append("offscreen")
                break
        print("SIGPATH", name, states)


A.run_program("re86", probe)
