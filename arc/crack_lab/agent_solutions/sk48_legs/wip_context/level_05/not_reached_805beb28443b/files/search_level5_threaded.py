"""Bounded macro beam search using the actually threaded target prefix."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


TARGET = (8, 9, 8)
ACTIONS = (1, 2, 3, 4)


def runs(spec):
    return tuple(action for action, count in spec for _ in range(count))


PREFIX = runs(
    (
        (4, 5), (1, 3), (2, 1), (3, 2), (2, 1), (6, 1),
        (1, 6), (2, 1), (4, 3), (6, 1), (1, 2), (4, 6), (2, 2),
        (2, 5), (3, 6), (4, 1), (3, 4), (1, 1), (2, 4),
        (6, 1), (3, 4), (4, 3), (6, 1), (4, 2), (6, 1), (1, 1),
    )
)
CONTROLLED_PAIR = (
    4, 4, 3, 3, 3, 3, 1, 1, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 2,
)
LOWER_FINAL_EIGHT = runs(
    (
        (1, 2), (2, 2), (4, 5), (1, 3), (2, 1), (3, 2),
        (2, 1), (1, 6), (2, 1), (4, 3), (1, 2), (4, 6), (2, 2),
    )
)
SUFFIX = runs(
    (
        (2, 5), (3, 6), (4, 1), (3, 4), (1, 1), (2, 4),
        (6, 1), (3, 4), (4, 3), (6, 1), (4, 2), (6, 1), (1, 1),
    )
)
SURPLUS_TRANSFER = (
    (3,) * 5
    + (1, 4, 1, 2, 2, 3)
    + (1,) * 4
    + (4,) * 7
    + (2,) * 2
)


def pieces(data):
    out = []
    for color in (8, 9):
        mask = data == color
        above = np.zeros_like(mask)
        left = np.zeros_like(mask)
        above[1:] = mask[:-1]
        left[:, 1:] = mask[:, :-1]
        for row, col in np.argwhere(mask & ~above & ~left):
            if row < 53:
                out.append((color, int(row), int(col)))
    return tuple(sorted(out, key=lambda piece: (piece[1], piece[2], piece[0])))


def observe(env):
    data = p.arr(env.frame())
    all_items = pieces(data)
    row = int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1
    scan_row = row + 1
    reach = 10
    while reach + 1 < 53 and int(data[scan_row, reach + 1]) not in (4, 5):
        reach += 1
    items = tuple(
        piece
        for piece in all_items
        if piece[1] == row and piece[2] <= reach
    )
    return (
        data,
        all_items,
        row,
        tuple(piece[0] for piece in items),
        reach,
        items,
    )


def target_prefix(sequence):
    count = 0
    for actual, wanted in zip(sequence, TARGET):
        if actual != wanted:
            break
        count += 1
    return count


def row_target_score(items):
    best = 0
    for row in {piece[1] for piece in items}:
        ordered = tuple(
            piece
            for piece in sorted(items, key=lambda piece: piece[2])
            if piece[1] == row
        )
        for start in range(len(ordered)):
            prefix = 0
            for piece, wanted in zip(ordered[start:], TARGET):
                if piece[0] != wanted:
                    break
                prefix += 1
            best = max(best, prefix)
    return best


def clear_target_geometry(items):
    best = 0
    for row in {piece[1] for piece in items if piece[1] != 25}:
        ordered = tuple(
            piece
            for piece in sorted(items, key=lambda piece: piece[2])
            if piece[1] == row
        )
        for start in range(len(ordered)):
            length = 0
            for index, (piece, wanted) in enumerate(
                zip(ordered[start:], TARGET)
            ):
                if piece[0] != wanted:
                    break
                if index and piece[2] - ordered[start + index - 1][2] != 6:
                    break
                length += 1
            best = max(best, length)
    return best


def clear_row_order_score(items):
    best = 0
    for row in {piece[1] for piece in items if piece[1] != 25}:
        ordered = tuple(
            piece[0]
            for piece in sorted(items, key=lambda piece: piece[2])
            if piece[1] == row
        )
        for start in range(len(ordered)):
            prefix = 0
            for color, wanted in zip(ordered[start:], TARGET):
                if color != wanted:
                    break
                prefix += 1
            best = max(best, prefix)
    return best


def score(observation):
    _data, items, row, sequence, reach, threaded_items = observation
    prefix = target_prefix(sequence)
    visible = row_target_score(items)
    clear_geometry = clear_target_geometry(items)
    clear_order = clear_row_order_score(items)
    clear_eights = sum(
        1 for color, piece_row, _col in items
        if color == 8 and piece_row != 25
    )
    subsequence = 0
    target_index = 0
    for color in sequence:
        if target_index < len(TARGET) and color == TARGET[target_index]:
            subsequence += 1
            target_index += 1

    geometry = 0
    if prefix:
        last_col = threaded_items[prefix - 1][2]
        candidates = [
            (color, piece_row, col)
            for color, piece_row, col in items
            if (color, piece_row, col) not in threaded_items[:prefix]
            and color == TARGET[prefix]
        ] if prefix < len(TARGET) else []
        if candidates:
            distance = min(
                abs(piece_row - row) + abs(col - (last_col + 6))
                for _color, piece_row, col in candidates
            )
            geometry = 120 - distance
    clear_lane = 20 if row != 25 else 0
    return (
        20000 * prefix
        + 1200 * subsequence
        + 250 * visible
        + 5000 * clear_geometry
        + 6000 * clear_order
        + 700 * clear_eights
        + 4 * geometry
        + clear_lane
        + reach,
        prefix,
        sequence,
        row,
        reach,
    )


def apply(env, actions):
    for action in actions:
        env.step(action)


def search(
    root,
    wanted_prefix=3,
    goal_fn=None,
    max_depth=18,
    beam_width=180,
):
    baseline = root.levels_completed
    frontier = [(root.clone(), (), None)]
    root_observation = observe(root)
    seen = {root_observation[0].tobytes()}
    best = (score(root_observation), ())
    print("ROOT", best[0], root_observation[1])

    for depth in range(1, max_depth + 1):
        previous_frontier = frontier
        candidates = []
        for parent_index, (node, path, last_action) in enumerate(previous_frontier):
            for action in ACTIONS:
                if action == last_action:
                    continue
                branch = node.clone()
                for run_count in range(1, 8):
                    try:
                        branch.step(action)
                    except IndexError:
                        break
                    child_path = path + (action,) * run_count
                    if branch.levels_completed > baseline:
                        return child_path, branch
                    observation = observe(branch)
                    if goal_fn is not None and goal_fn(branch, observation):
                        return child_path, branch
                    if target_prefix(observation[3]) >= wanted_prefix:
                        return child_path, branch
                    fingerprint = observation[0].tobytes()
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    value = score(observation)
                    if value > best[0]:
                        best = (value, child_path)
                        print(
                            "BEST",
                            depth,
                            value,
                            list(child_path),
                            observation[1],
                        )
                    candidates.append(
                        (
                            value,
                            child_path,
                            action,
                            (
                                observation[1],
                                observation[2],
                                observation[3],
                                observation[4],
                            ),
                            parent_index,
                            run_count,
                        )
                    )

        candidates.sort(
            key=lambda item: (item[0][0], -len(item[1])),
            reverse=True,
        )
        configuration_counts = {}
        prefix_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        prefix_quota = max(12, beam_width // 5)
        deferred = []
        selected = []
        for (
            value,
            path,
            action,
            configuration,
            parent_index,
            run_count,
        ) in candidates:
            prefix = value[1]
            configuration_count = configuration_counts.get(
                (configuration, prefix), 0
            )
            if configuration_count >= 3:
                continue
            if prefix_counts[prefix] >= prefix_quota:
                deferred.append(
                    (
                        value,
                        path,
                        action,
                        configuration,
                        parent_index,
                        run_count,
                    )
                )
                continue
            configuration_counts[(configuration, prefix)] = (
                configuration_count + 1
            )
            prefix_counts[prefix] += 1
            selected.append((path, action, parent_index, run_count))
        for (
            value,
            path,
            action,
            configuration,
            parent_index,
            run_count,
        ) in deferred:
            if len(selected) >= beam_width:
                break
            prefix = value[1]
            configuration_count = configuration_counts.get(
                (configuration, prefix), 0
            )
            if configuration_count >= 3:
                continue
            configuration_counts[(configuration, prefix)] = (
                configuration_count + 1
            )
            selected.append((path, action, parent_index, run_count))
        frontier = []
        for path, action, parent_index, run_count in selected[:beam_width]:
            parent_node = previous_frontier[parent_index][0]
            child = parent_node.clone()
            apply(child, (action,) * run_count)
            frontier.append((child, path, action))
        print(
            "DEPTH",
            depth,
            "frontier", len(frontier),
            "seen", len(seen),
            "prefixes", prefix_counts,
        )
        if not frontier:
            break
    return None, None


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(
        env,
        PREFIX
        + CONTROLLED_PAIR
        + LOWER_FINAL_EIGHT
        + (3, 3),
    )
    path, node = search(
        env,
        wanted_prefix=99,
        goal_fn=lambda node, observation: (
            node.levels_completed > 4
            or clear_row_order_score(observation[1]) >= 3
        ),
    )
    print("SEARCH", list(path) if path is not None else None)
    if node is not None:
        observation = observe(node)
        print("FOUND", node.levels_completed, score(observation), observation[1])


if __name__ == "__main__":
    levels, path, error = arena.run_program("sk48", run)
    print("END", levels, len(path), error)
