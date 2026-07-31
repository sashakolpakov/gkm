"""Temporary compact level-5 observation probe, run only through gkm_try.py."""
import numpy as np

import perception as p
import legs


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
RECOLLECT_PAIR_LEFT = (
    2, 2, 3, 3, 1, 1, 1, 4, 4,
    3, 3, 3, 3, 3, 3, 1, 4, 4, 4, 3, 3, 3, 2,
)


def pieces(env):
    return tuple(
        (b.color, b.bbox[0], b.bbox[1])
        for b in p.connected_components(env.frame(), colors=(8, 9), min_area=4)
        if b.bbox[0] < 53
    )


def summary(env):
    data = p.arr(env.frame())
    avatar_row = int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1
    items = pieces(env)
    before = set(items)
    controlled = ()
    for action in (3, 4):
        branch = env.clone()
        branch.step(action)
        removed = before - set(pieces(branch))
        for row in {item[1] for item in removed}:
            seq = tuple(
                color for color, _row, _col in sorted(
                    (item for item in removed if item[1] == row),
                    key=lambda item: item[2],
                )
            )
            if len(seq) > len(controlled):
                controlled = seq
    return env.levels_completed, avatar_row, controlled, items


def analyze(env):
    data = p.arr(env.frame())
    items = pieces(env)
    before = set(items)
    controlled_items = ()
    for action in (3, 4):
        branch = env.clone()
        if branch.terminal():
            continue
        branch.step(action)
        removed = before - set(pieces(branch))
        for row in {item[1] for item in removed}:
            group = tuple(sorted(
                (item for item in removed if item[1] == row),
                key=lambda item: item[2],
            ))
            if len(group) > len(controlled_items):
                controlled_items = group
    sequence = tuple(item[0] for item in controlled_items)
    prefix = 0
    for actual, wanted in zip(sequence, (8, 9, 8)):
        if actual != wanted:
            break
        prefix += 1
    row = int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1
    return items, controlled_items, prefix, row


def search(root, max_actions=39, max_segments=10, beam_width=240):
    baseline = root.levels_completed
    frontier = [(root.clone(), (), None)]
    seen = {p.arr(root.frame()).tobytes()}
    best = None
    for segment in range(max_segments):
        candidates = []
        for node, path, last_action in frontier:
            for action in (1, 2, 3, 4, 6):
                if action == last_action:
                    continue
                branch = node.clone()
                for count in range(1, 7):
                    if len(path) + count > max_actions or branch.terminal():
                        break
                    branch.step(action)
                    child_path = path + (action,) * count
                    if branch.levels_completed > baseline:
                        print("FOUND_PATH", child_path)
                        return child_path
                    fingerprint = p.arr(branch.frame()).tobytes()
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    items, controlled, prefix, avatar_row = analyze(branch)
                    if controlled:
                        tail = controlled[-1]
                        remaining = [item for item in items if item not in controlled]
                        desired = 8 if prefix == 2 else (9 if prefix == 1 else 8)
                        distances = [
                            abs(r - tail[1]) + abs(c - (tail[2] + 6))
                            for color, r, c in remaining if color == desired
                        ]
                        distance = min(distances, default=100)
                        right = tail[2]
                    else:
                        distance, right = 100, 0
                    clear = 1 if avatar_row != 25 else 0
                    beyond = 1 if controlled and controlled[0][2] >= 36 else 0
                    value = (
                        100000 * prefix
                        + 8000 * beyond
                        + 2000 * clear
                        + 80 * right
                        - 100 * distance
                        - len(child_path)
                    )
                    key = (value, prefix, beyond, clear)
                    if best is None or key > best[0]:
                        best = (key, child_path, (avatar_row, controlled, items))
                        print("SEARCH_BEST", segment + 1, best)
                    candidates.append(
                        (value, prefix, beyond, clear, -len(child_path),
                         child_path, branch.clone(), action)
                    )
        candidates.sort(reverse=True, key=lambda item: item[:5])
        frontier = []
        quotas = {}
        for item in candidates:
            configuration = (item[1], item[2], item[3])
            used = quotas.get(configuration, 0)
            if used >= max(20, beam_width // 3):
                continue
            quotas[configuration] = used + 1
            frontier.append((item[6], item[5], item[7]))
            if len(frontier) >= beam_width:
                break
        print("SEARCH_DEPTH", segment + 1, len(frontier), len(seen))
        if not frontier:
            break
    print("SEARCH_NONE", best)
    return None


def minimize_pair(root, path):
    candidate = tuple(path)

    def valid(trial):
        node = p.replay(root, trial)
        return not node.terminal() and analyze(node)[2] >= 2

    granularity = 2
    while len(candidate) >= 2:
        chunk = max(1, (len(candidate) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(candidate), chunk):
            trial = candidate[:start] + candidate[start + chunk:]
            if valid(trial):
                candidate = trial
                print("REDUCE", len(candidate), start, chunk)
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if not reduced:
            if chunk == 1:
                break
            granularity = min(len(candidate), granularity * 2)
    return candidate


def fast_score(env):
    items = pieces(env)
    best = 0
    for row in {item[1] for item in items}:
        ordered = tuple(sorted(
            (item for item in items if item[1] == row),
            key=lambda item: item[2],
        ))
        for start in range(len(ordered)):
            prefix = 0
            geometry = 0
            for index, (item, wanted) in enumerate(
                zip(ordered[start:], (8, 9, 8))
            ):
                if item[0] != wanted:
                    break
                prefix += 1
                if index and item[2] - ordered[start + index - 1][2] == 6:
                    geometry += 1
            best = max(best, 20000 * prefix + 5000 * geometry)
    contacts = sum(
        1
        for a in items
        for b in items
        if a < b and a[2] == b[2] and abs(a[1] - b[1]) == 6
    )
    spread = len({item[1] for item in items})
    return best + 500 * contacts + 80 * spread, items


def fast_search(root, max_actions=90, max_segments=15, beam_width=160):
    baseline = root.levels_completed
    frontier = [((), None)]
    seen = {p.arr(root.frame()).tobytes()}
    best = None
    for depth in range(1, max_segments + 1):
        candidates = []
        for path, last_action in frontier:
            node = p.replay(root, path)
            for action in (1, 2, 3, 4, 6):
                if action == last_action:
                    continue
                branch = node.clone()
                for count in range(1, 7):
                    if len(path) + count > max_actions or branch.terminal():
                        break
                    branch.step(action)
                    child_path = path + (action,) * count
                    if branch.levels_completed > baseline:
                        print("FAST_FOUND", child_path)
                        return child_path
                    fingerprint = p.arr(branch.frame()).tobytes()
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    value, items = fast_score(branch)
                    record = (value - len(child_path), child_path, action, items)
                    if best is None or record[0] > best[0]:
                        best = record
                        print("FAST_BEST", depth, best)
                    candidates.append(record)
        candidates.sort(reverse=True, key=lambda item: item[0])
        frontier = []
        configuration_counts = {}
        for value, path, action, items in candidates:
            count = configuration_counts.get(items, 0)
            if count >= 3:
                continue
            configuration_counts[items] = count + 1
            frontier.append((path, action))
            if len(frontier) >= beam_width:
                break
        print("FAST_DEPTH", depth, len(frontier), len(seen))
        if not frontier:
            break
    print("FAST_NONE", best)
    return None


def state(env):
    objects = p.object_candidates(env.frame())
    avatar = tuple(
        (o["color"], o["bbox"], o["area"])
        for o in objects
        if o["color"] == 6 and o["bbox"][0] < 53
    )
    pieces = tuple(
        (o["color"], o["bbox"], o["area"])
        for o in objects
        if o["color"] in (8, 9, 14)
    )
    return avatar, pieces


def run(env):
    pair_path = (
        1, 3, 3, 4, 4, 4, 1, 4, 4, 2, 3, 3, 3, 3, 2, 2, 2,
        3, 3, 4, 4, 4, 4, 4, 1, 4, 4, 3, 3, 3, 3, 1, 1, 4,
    )
    root = p.replay(env, pair_path)
    print("FAST_ROOT", summary(root))
    fast_search(root)
