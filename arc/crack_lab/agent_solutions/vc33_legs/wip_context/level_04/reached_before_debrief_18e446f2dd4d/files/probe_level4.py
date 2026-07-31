import importlib.util
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import (
    arr,
    bounded_replay_bfs,
    color_counts,
    connected_components,
    frame_delta,
)
from legs import align_lower_cyan_tip_with_upper_notch


spec = importlib.util.spec_from_file_location("solve", "solve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=2)
        if b.area < 1500
    ]


def column_runs(frame, columns=(3, 8, 18, 24, 33, 39, 48, 51, 58, 62)):
    data = arr(frame)
    out = {}
    for column in columns:
        runs = []
        start = 1
        color = int(data[1, column])
        for row in range(2, 61):
            value = int(data[row, column])
            if value != color:
                runs.append((start, row - 1, color))
                start, color = row, value
        runs.append((start, 60, color))
        out[column] = tuple(runs)
    return out


def hydraulic_state(env):
    data = arr(env.frame())
    markers = connected_components(data, colors=(11,), min_area=2)
    agents = connected_components(data, colors=(1,), min_area=2)
    return (
        tuple(b.bbox for b in markers),
        tuple(int((data[1:, column] == 0).sum()) for column in (8, 24, 33, 48, 62)),
        tuple(b.bbox for b in agents),
    )


def inspect(env):
    module.solve(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    print("RUNS", column_runs(env.frame()))
    print("BLOBS", compact_blobs(env.frame()))
    controls = connected_components(env.frame(), colors=(9,), min_area=2)
    for index, control in enumerate(controls):
        x = int(control.centroid[1])
        y = int(control.centroid[0])
        clone = env.clone()
        before = clone.frame()
        clone.step(6, x, y)
        delta = frame_delta(before, clone.frame())
        print(
            "CLICK",
            index,
            (x, y),
            "LEVEL",
            clone.levels_completed,
            "DELTA",
            (delta["count"], delta["bbox"]),
            "BLOBS",
            compact_blobs(clone.frame()),
        )
        print("CLICK_RUNS", index, column_runs(clone.frame()))
    for index, control in enumerate(controls):
        x = int(control.centroid[1])
        y = int(control.centroid[0])
        clone = env.clone()
        trace = []
        for press in range(1, 9):
            clone.step(6, x, y)
            markers = connected_components(
                clone.frame(), colors=(11,), min_area=2
            )
            trace.append(
                (
                    press,
                    clone.levels_completed,
                    tuple((b.bbox, b.area) for b in markers),
                )
            )
            if clone.levels_completed > env.levels_completed:
                break
        print("REPEAT", index, (x, y), trace)
    attempt = env.clone()
    align_lower_cyan_tip_with_upper_notch(
        attempt, max_presses=20, marker_color=11, max_states=300
    )
    print(
        "ALIGN_ATTEMPT",
        attempt.levels_completed,
        tuple(
            (b.bbox, b.area)
            for b in connected_components(
                attempt.frame(), colors=(11,), min_area=2
            )
        ),
    )
    progress = env.clone()
    progress_path = []

    def gap(node):
        markers = connected_components(node.frame(), colors=(11,), min_area=2)
        if len(markers) != 2:
            return None
        return abs(markers[0].centroid[0] - markers[1].centroid[0])

    def actions(node):
        return [
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(node.frame(), colors=(9,), min_area=2)
        ]

    while gap(progress) and len(progress_path) < 40:
        before_gap = gap(progress)
        path = bounded_replay_bfs(
            progress,
            lambda node, _: (
                node.levels_completed > env.levels_completed
                or (gap(node) is not None and gap(node) < before_gap)
            ),
            actions,
            key_fn=lambda node: (
                node.levels_completed,
                arr(node.frame())[1:].tobytes(),
            ),
            max_states=300,
            max_depth=10,
        )
        print("PROGRESS", before_gap, path)
        if not path:
            break
        for action in path:
            progress.step(*action)
        progress_path.extend(path)
        if progress.levels_completed > env.levels_completed:
            break
    print(
        "PROGRESS_RESULT",
        progress.levels_completed,
        gap(progress),
        progress_path,
    )
    for name, indexes in (
        ("LEFT_RELAY", (2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4)),
        ("RIGHT_RELAY", (3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5)),
        ("EMPTY_B", (3, 3, 3, 3, 3, 3)),
        ("FILL_B", (2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4)),
    ):
        clone = env.clone()
        trace = []
        for index in indexes:
            control = controls[index]
            action = (6, int(control.centroid[1]), int(control.centroid[0]))
            clone.step(*action)
            trace.append((index, hydraulic_state(clone)))
        print("PATTERN", name, trace)
    candidates = [
        b
        for b in connected_components(env.frame(), min_area=2)
        if b.color in (1, 4, 5, 11)
    ]
    for blob in candidates:
        x, y = int(blob.centroid[1]), int(blob.centroid[0])
        clone = env.clone()
        before = clone.frame()
        clone.step(6, x, y)
        delta = frame_delta(before, clone.frame())
        outside_header = int((arr(before)[1:] != arr(clone.frame())[1:]).sum())
        print(
            "OBJECT_CLICK",
            (blob.color, blob.bbox, x, y),
            "PIXELS",
            outside_header,
            "DELTA",
            (delta["count"], delta["bbox"]),
            "STATE",
            hydraulic_state(clone),
        )
    root = env.clone()
    queue = deque([(root, [])])
    seen = {arr(root.frame())[1:].tobytes()}
    found = None
    context_effects = []
    while queue and len(seen) <= 1200:
        node, path = queue.popleft()
        node_key = arr(node.frame())[1:].tobytes()
        node_candidates = [
            b
            for b in connected_components(node.frame(), min_area=2)
            if b.color != 9 and b.color != 7 and b.area < 1500
        ]
        for blob in node_candidates:
            child = node.clone()
            child.step(
                6, int(blob.centroid[1]), int(blob.centroid[0])
            )
            child_key = arr(child.frame())[1:].tobytes()
            if (
                child.levels_completed > env.levels_completed
                or child_key != node_key
            ):
                context_effects.append(
                    (
                        path,
                        blob.color,
                        blob.bbox,
                        child.levels_completed,
                        hydraulic_state(child),
                    )
                )
                if len(context_effects) >= 8:
                    queue.clear()
                    break
        if len(context_effects) >= 8:
            break
        if len(path) >= 24:
            continue
        node_controls = connected_components(
            node.frame(), colors=(9,), min_area=2
        )
        for index, control in enumerate(node_controls):
            child = node.clone()
            child.step(
                6, int(control.centroid[1]), int(control.centroid[0])
            )
            child_path = path + [index]
            if child.levels_completed > env.levels_completed:
                found = child_path
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("BUTTON_BFS", len(seen), found, "CONTEXT", context_effects)
    interactive = [
        b
        for b in connected_components(env.frame(), min_area=2)
        if b.color in (1, 4, 11)
    ]
    for blob in interactive:
        object_action = (
            6,
            int(blob.centroid[1]),
            int(blob.centroid[0]),
        )
        for index, control in enumerate(controls):
            control_action = (
                6,
                int(control.centroid[1]),
                int(control.centroid[0]),
            )
            direct = env.clone()
            direct.step(*control_action)
            selected = env.clone()
            selected.step(*object_action)
            selected.step(*control_action)
            if (
                selected.levels_completed != direct.levels_completed
                or arr(selected.frame())[1:].tobytes()
                != arr(direct.frame())[1:].tobytes()
            ):
                print(
                    "SELECT_THEN_BUTTON",
                    (blob.color, blob.bbox),
                    index,
                    hydraulic_state(selected),
                )
            after_button = env.clone()
            after_button.step(*control_action)
            baseline = arr(after_button.frame())[1:].tobytes()
            after_button.step(*object_action)
            if (
                after_button.levels_completed != env.levels_completed
                or arr(after_button.frame())[1:].tobytes() != baseline
            ):
                print(
                    "BUTTON_THEN_OBJECT",
                    index,
                    (blob.color, blob.bbox),
                    hydraulic_state(after_button),
                )
    for first in interactive:
        for second in interactive:
            clone = env.clone()
            clone.step(6, int(first.centroid[1]), int(first.centroid[0]))
            before = arr(clone.frame())[1:].tobytes()
            clone.step(6, int(second.centroid[1]), int(second.centroid[0]))
            if (
                clone.levels_completed != env.levels_completed
                or arr(clone.frame())[1:].tobytes() != before
            ):
                print(
                    "OBJECT_PAIR",
                    (first.color, first.bbox),
                    (second.color, second.bbox),
                    hydraulic_state(clone),
                )
    staged = env.clone()
    staged_path = []

    def avatar(node):
        markers = connected_components(node.frame(), colors=(11,), min_area=2)
        movable = [b for b in markers if b.area >= 12]
        return movable[0] if movable else None

    def target(node):
        markers = connected_components(node.frame(), colors=(11,), min_area=2)
        fixed = [b for b in markers if b.area < 12]
        return fixed[0] if fixed else None

    def dynamic_actions(node):
        options = [
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(
                node.frame(), colors=(9,), min_area=2
            )
        ]
        options.extend(
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(node.frame(), min_area=2)
            if b.color in (12, 13, 14, 15)
        )
        return options

    for stage in range(20):
        moving = avatar(staged)
        fixed = target(staged)
        if moving is None or fixed is None:
            break
        start_col = moving.centroid[1]
        start_gap = abs(moving.centroid[0] - fixed.centroid[0])

        def dense_goal(node):
            if node.levels_completed > env.levels_completed:
                return True
            current = avatar(node)
            goal = target(node)
            if current is None or goal is None:
                return False
            if current.centroid[1] > start_col:
                return True
            return (
                start_col >= goal.centroid[1]
                and
                current.centroid[1] == start_col
                and abs(current.centroid[0] - goal.centroid[0]) < start_gap
            )

        queue = deque([(staged.clone(), [])])
        local_seen = {arr(staged.frame())[1:].tobytes()}
        stage_path = None
        while queue and len(local_seen) <= 900:
            node, path = queue.popleft()
            if len(path) >= 16:
                continue
            for action in dynamic_actions(node):
                child = node.clone()
                child.step(*action)
                child_path = path + [action]
                if dense_goal(child):
                    stage_path = child_path
                    queue.clear()
                    break
                key = arr(child.frame())[1:].tobytes()
                if key not in local_seen:
                    local_seen.add(key)
                    queue.append((child, child_path))
        print(
            "STAGE",
            stage,
            (start_col, start_gap),
            len(local_seen),
            stage_path,
        )
        if not stage_path:
            break
        for action in stage_path:
            staged.step(*action)
        staged_path.extend(stage_path)
        if staged.levels_completed > env.levels_completed:
            break
    print(
        "STAGED_RESULT",
        staged.levels_completed,
        hydraulic_state(staged),
        staged_path,
    )


levels, path, err = A.run_program("vc33", inspect)
print("DONE", levels, len(path), err)
