"""Bounded beam search from the verified visual 8-9-8 level-5 state."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


TARGET = (8, 9, 8)
PREFIX_RUNS = (
    (4, 5), (1, 3), (2, 1), (3, 2), (2, 1), (6, 1),
    (1, 6), (2, 1), (4, 3), (6, 1), (1, 2), (4, 6), (2, 2),
)
SUFFIX_RUNS = (
    (2, 5), (3, 6), (4, 1), (3, 4), (1, 1), (2, 4),
    (6, 1), (3, 4), (4, 3), (6, 1), (4, 2), (6, 1), (1, 1),
)


def runs(spec):
    return tuple(action for action, count in spec for _ in range(count))


PREFIX = runs(PREFIX_RUNS) + runs(SUFFIX_RUNS)
CONTROLLED_PAIR = (
    4, 4,
    3, 3, 3, 3,
    1, 1,
    4, 4, 4, 4,
    3, 3, 3, 3, 3,
    2,
)
PAIR_LADDER = (
    4, 4, 4, 4, 4,
    1, 1, 1,
    2,
    3,
    2,
    1, 1, 1, 1, 1, 1,
    2,
    4, 4, 4,
    1, 1,
    4, 4, 4,
    2, 2,
)
LOWER_FINAL_EIGHT = (
    1, 1,
    2, 2,
    4, 4, 4, 4, 4,
    1, 1, 1,
    2,
    3, 3,
    2,
    1, 1, 1, 1, 1, 1,
    2,
    4, 4, 4,
    1, 1,
    4, 4, 4, 4, 4, 4,
    2, 2,
)
RECOLLECT_PAIR_LEFT = (
    2, 2,
    3, 3,
    1, 1, 1,
    4, 4,
    3, 3, 3, 3, 3, 3,
    1,
    4, 4, 4,
    3, 3, 3,
    2,
)


def pieces(env):
    data = p.arr(env.frame())
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
    return tuple(sorted(out))


def target_rows(items):
    found = []
    for row in sorted({item[1] for item in items}):
        ordered = sorted(
            ((col, color) for color, piece_row, col in items if piece_row == row)
        )
        for index in range(len(ordered) - 2):
            window = ordered[index:index + 3]
            if (
                tuple(color for _col, color in window) == TARGET
                and window[1][0] - window[0][0] == 6
                and window[2][0] - window[1][0] == 6
            ):
                found.append((row, window[0][0]))
    return tuple(found)


def score(env):
    items = pieces(env)
    before = set(items)
    controlled = ()
    for action in (3, 4):
        branch = env.clone()
        try:
            branch.step(action)
        except IndexError:
            continue
        removed = before - set(pieces(branch))
        for row in {item[1] for item in removed}:
            sequence = tuple(
                color
                for color, _row, _col in sorted(
                    (item for item in removed if item[1] == row),
                    key=lambda item: item[2],
                )
            )
            if len(sequence) > len(controlled):
                controlled = sequence
    control_prefix = 0
    for actual, wanted in zip(controlled, TARGET):
        if actual != wanted:
            break
        control_prefix += 1
    data = p.arr(env.frame())
    avatar_row = int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1
    sequence = tuple(
        color
        for color, _row, _col in sorted(
            (item for item in items if item[1] == avatar_row),
            key=lambda item: item[2],
        )
    )
    prefix = 0
    for actual, wanted in zip(sequence, TARGET):
        if actual != wanted:
            break
        prefix += 1
    subsequence = 0
    target_index = 0
    for color in sequence:
        if target_index < len(TARGET) and color == TARGET[target_index]:
            subsequence += 1
            target_index += 1
    vertical_contacts = sum(
        1
        for color_a, row_a, col_a in items
        for color_b, row_b, col_b in items
        if (color_a, row_a, col_a) < (color_b, row_b, col_b)
        and col_a == col_b
        and abs(row_a - row_b) == 6
    )
    return (
        50000 * control_prefix
        + 10000 * prefix
        + 1000 * subsequence
        - 100 * abs(len(sequence) - 3)
        + 40 * len(target_rows(items))
        + 30 * vertical_contacts
        + max((col for _color, _row, col in items), default=0),
        avatar_row,
        sequence,
        controlled,
    )


def dynamic_key(env, last_action):
    data = p.arr(env.frame())
    dynamic = np.where(np.isin(data, (0, 2, 3, 6, 8, 9)), data, 0)
    return dynamic.tobytes(), last_action


def search(root, max_depth=18, beam_width=120):
    baseline = root.levels_completed
    frontier = [(root.clone(), (), None)]
    seen = {dynamic_key(root, None)}
    best = (score(root), ())
    print("ROOT", best[0], pieces(root))
    for depth in range(1, max_depth + 1):
        candidates = []
        for node, path, last_action in frontier:
            for action in (1, 2, 3, 4):
                if action == last_action:
                    continue
                branch = node.clone()
                for count in range(1, 7):
                    try:
                        branch.step(action)
                    except IndexError:
                        break
                    child_path = path + (action,) * count
                    if branch.levels_completed > baseline:
                        return child_path, branch
                    key = dynamic_key(branch, action)
                    if key in seen:
                        continue
                    seen.add(key)
                    value = score(branch)
                    if value > best[0]:
                        best = (value, child_path)
                        print(
                            "BEST",
                            depth,
                            value,
                            list(child_path),
                            pieces(branch),
                        )
                    candidates.append((value, child_path, branch.clone(), action))
        candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        frontier = []
        configuration_counts = {}
        for _value, path, node, action in candidates:
            configuration = pieces(node)
            count = configuration_counts.get(configuration, 0)
            if count >= 4:
                continue
            configuration_counts[configuration] = count + 1
            frontier.append((node, path, action))
            if len(frontier) == beam_width:
                break
        print("DEPTH", depth, "frontier", len(frontier), "seen", len(seen))
        if not frontier:
            break
    return None, None


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    root = p.replay(
        env,
        PREFIX
        + CONTROLLED_PAIR
        + LOWER_FINAL_EIGHT
        + runs(SUFFIX_RUNS)
        + RECOLLECT_PAIR_LEFT,
    )
    path, node = search(root)
    print("SEARCH", list(path) if path else None)
    if node is not None:
        print("FOUND", node.levels_completed, pieces(node))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
