import itertools
import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


DIRECTIONS = (
    (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0),
    (0, 2), (1, -1), (1, 0), (1, 1),
)
VELOCITY = {
    (-2, 0): (-1, 0),
    (-1, -1): (-1, -1),
    (-1, 1): (-1, 1),
    (0, -2): (0, -1),
    (0, 0): None,
    (0, 2): (0, 1),
    (1, -1): (1, -1),
    (1, 0): (1, 0),
    (1, 1): (1, 1),
}
ROW_BOUNDS = (12, 62)
COL_BOUNDS = (2, 61)


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def advance(value, velocity, bounds):
    candidate = value + 4 * velocity
    if candidate < bounds[0] or candidate > bounds[1]:
        velocity = -velocity
        candidate = value + 4 * velocity
    return candidate, velocity


def transition(state, selected, offset):
    result = []
    chosen_velocity = VELOCITY[offset]
    for index, (row, col, velocity_row, velocity_col) in enumerate(state):
        if index == selected:
            if chosen_velocity is None:
                result.append((row, col, velocity_row, velocity_col))
                continue
            direction_row, direction_col = chosen_velocity
            distance = 12 if 0 in chosen_velocity else 8
            row = min(ROW_BOUNDS[1], max(
                ROW_BOUNDS[0], row + distance * direction_row
            ))
            col = min(COL_BOUNDS[1], max(
                COL_BOUNDS[0], col + distance * direction_col
            ))
            if direction_row:
                velocity_row = -direction_row
            if direction_col:
                velocity_col = -direction_col
        else:
            row, velocity_row = advance(
                row, velocity_row, ROW_BOUNDS
            )
            col, velocity_col = advance(
                col, velocity_col, COL_BOUNDS
            )
        result.append((row, col, velocity_row, velocity_col))
    return tuple(result)


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    masks = tuple(
        {
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        }
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )[1:]
    root = env.clone()
    for action in PREFIX:
        root.step(*action)
    root_groups = body_groups(root.frame())
    offsets = tuple(
        (row - center(root_groups[0])[0], col - center(root_groups[0])[1])
        for row, col in root_groups[0]
    )
    valid = tuple(
        {
            (row, col)
            for row in range(ROW_BOUNDS[0], ROW_BOUNDS[1] + 1)
            for col in range(COL_BOUNDS[0], COL_BOUNDS[1] + 1)
            if all((row + dr, col + dc) in mask for dr, dc in offsets)
        }
        for mask in masks
    )
    initial_state = (
        (19, 57, 0, -1),
        (52, 53, -1, -1),
        (54, 7, -1, 0),
    )

    def goal(state):
        positions = tuple((item[0], item[1]) for item in state)
        return any(
            all(position in valid[target]
                for position, target in zip(positions, order))
            for order in itertools.permutations(range(3))
        )

    queue = deque([initial_state])
    parents = {initial_state: None}
    goals = []
    while queue and len(parents) < 500000 and len(goals) < 40:
        state = queue.popleft()
        parent = parents[state]
        depth = 0 if parent is None else parent[2] + 1
        if depth >= 16:
            continue
        for selected in range(3):
            for offset in DIRECTIONS:
                child = transition(state, selected, offset)
                if child in parents:
                    continue
                parents[child] = (state, (selected, offset), depth)
                if goal(child):
                    goals.append(child)
                queue.append(child)
    print("MODEL", len(parents), "goals", len(goals), flush=True)

    for goal_state in goals:
        controls = []
        state = goal_state
        while parents[state] is not None:
            prior, control, _ = parents[state]
            controls.append(control)
            state = prior
        controls.reverse()
        node = root.clone()
        actions = []
        predicted = initial_state
        matched = True
        mismatch = None
        for selected, offset in controls:
            row, col = predicted[selected][:2]
            if (row, col) not in tuple(
                center(group) for group in body_groups(node.frame())
            ):
                matched = False
                break
            action = (6, col + offset[1], row + offset[0])
            node.step(*action)
            actions.append(action)
            predicted = transition(predicted, selected, offset)
            actual = tuple(sorted(
                center(group) for group in body_groups(node.frame())
            ))
            expected = tuple(sorted(
                (item[0], item[1]) for item in predicted
            ))
            if actual != expected:
                matched = False
                mismatch = (action, expected, actual)
                break
        print(
            "TRY", len(controls), "matched", matched,
            "level", int(node.levels_completed),
            "actions", actions, "mismatch", mismatch, flush=True,
        )
        if int(node.levels_completed) > start_level:
            print("FOUND", PREFIX + actions, flush=True)
            return


A.run_program("su15", inspect)
