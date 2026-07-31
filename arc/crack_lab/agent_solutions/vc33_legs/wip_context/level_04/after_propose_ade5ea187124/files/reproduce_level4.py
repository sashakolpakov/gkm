import importlib.util
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def center(blob):
    return (6, int(blob.centroid[1]), int(blob.centroid[0]))


def compact(frame, colors=None):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=colors, min_area=2)
        if blob.area < 1000
    ]


def avatar(env):
    markers = connected_components(env.frame(), colors=(11,), min_area=2)
    movable = [blob for blob in markers if blob.area >= 12]
    return movable[0] if movable else None


def target(env):
    markers = connected_components(env.frame(), colors=(11,), min_area=2)
    fixed = [blob for blob in markers if blob.area < 12]
    return fixed[0] if fixed else None


def available_actions(env):
    actions = []
    for index, blob in enumerate(
        connected_components(env.frame(), colors=(9,), min_area=2)
    ):
        actions.append((center(blob), ("button", index)))
    for blob in connected_components(
        env.frame(), colors=(12, 13, 14, 15), min_area=2
    ):
        actions.append((center(blob), ("gate", blob.bbox)))
    return actions


def progress(env):
    moving, fixed = avatar(env), target(env)
    if moving is None or fixed is None:
        return None
    remaining_gates = [
        blob
        for blob in connected_components(
            env.frame(), colors=(1, 12, 13, 14, 15), min_area=2
        )
        if moving.centroid[1] < blob.centroid[1] < fixed.centroid[1]
    ]
    return (
        int(moving.centroid[1]),
        len(remaining_gates),
        int(abs(moving.centroid[0] - fixed.centroid[0])),
    )


def inspect(env):
    solver.solve(env)
    base_level = env.levels_completed
    print("ROOT", base_level, env.actions)
    print("OBJECTS", compact(env.frame()))

    controls = connected_components(env.frame(), colors=(9,), min_area=2)
    print("CONTROLS", [center(blob)[1:] for blob in controls])
    for index, control in enumerate(controls):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(*center(control))
        delta = frame_delta(before, clone.frame())
        print(
            "DIRECT",
            index,
            (delta["count"], delta["bbox"]),
            compact(clone.frame(), colors=(1, 11, 12)),
        )

    root = env.clone()
    queue = deque([(root, ())])
    seen = {arr(root.frame())[1:].tobytes()}
    found = None
    while queue and len(seen) <= 100 and found is None:
        node, path = queue.popleft()
        before_key = arr(node.frame())[1:].tobytes()
        for gate in connected_components(node.frame(), colors=(1, 12), min_area=2):
            child = node.clone()
            child.step(*center(gate))
            after_key = arr(child.frame())[1:].tobytes()
            if child.levels_completed > base_level or after_key != before_key:
                found = (
                    path,
                    (gate.color, gate.bbox),
                    child.levels_completed,
                    compact(child.frame(), colors=(1, 11, 12)),
                )
                break
        if found is not None or len(path) >= 4:
            continue
        for index, control in enumerate(
            connected_components(node.frame(), colors=(9,), min_area=2)
        ):
            child = node.clone()
            child.step(*center(control))
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (index,)))
    print("GATE_SEARCH", len(seen), found)

    staged = env.clone()
    full_path = []
    for stage in range(8):
        initial = progress(staged)
        if initial is None or staged.levels_completed > base_level:
            break
        start_column, gates_left, start_gap = initial

        def improved(node):
            if node.levels_completed > base_level:
                return True
            current = progress(node)
            if current is None:
                return False
            column, remaining, gap = current
            if gates_left:
                return column > start_column
            return column == start_column and remaining == 0 and gap < start_gap

        queue = deque([(staged.clone(), [], [])])
        local_seen = {arr(staged.frame())[1:].tobytes()}
        result = None
        while queue and len(local_seen) <= 700 and result is None:
            node, path, labels = queue.popleft()
            if len(path) >= 16:
                continue
            for action, label in available_actions(node):
                child = node.clone()
                child.step(*action)
                child_path = path + [action]
                child_labels = labels + [label]
                if improved(child):
                    result = (child_path, child_labels)
                    break
                key = arr(child.frame())[1:].tobytes()
                if key not in local_seen:
                    local_seen.add(key)
                    queue.append((child, child_path, child_labels))
        print("STAGE", stage, initial, len(local_seen), None if result is None else result[1])
        if result is None:
            break
        for action in result[0]:
            staged.step(*action)
        full_path.extend(result[1])
    print("STAGED", staged.levels_completed, progress(staged), full_path)


levels, path, error = arena.run_program("vc33", inspect)
print("DONE", levels, len(path), error)
