"""Bounded macro search for a verified controlled 8 prefix on sk48 level 5."""
import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


TARGET = (8, 9, 8)


def pieces(env):
    out = []
    for blob in p.connected_components(env.frame(), colors=(8, 9), min_area=4):
        if blob.bbox[0] < 53:
            out.append((blob.color, blob.bbox[0], blob.bbox[1]))
    return out


def avatar_top(env):
    data = p.arr(env.frame())
    return int(np.where(data[:53, :11] == 6)[0].min())


def row_sequence(env):
    top = avatar_top(env) + 1
    return tuple(color for color, row, _col in sorted(pieces(env), key=lambda x: x[2])
                 if row == top)


def lcs_target(sequence):
    best = [0] * (len(TARGET) + 1)
    for color in sequence:
        previous = best[:]
        for j, wanted in enumerate(TARGET, start=1):
            if color == wanted:
                best[j] = max(best[j], previous[j - 1] + 1)
            else:
                best[j] = max(best[j], best[j - 1])
    return best[-1]


def score(env):
    sequence = row_sequence(env)
    prefix = 0
    for actual, wanted in zip(sequence, TARGET):
        if actual != wanted:
            break
        prefix += 1
    return (
        prefix * 1000
        + lcs_target(sequence) * 80
        + sequence.count(8) * 25
        - abs(len(sequence) - 3) * 8
    )


def key(env):
    return p.arr(env.frame()).tobytes()


def controlled_eight_prefix(env):
    sequence = row_sequence(env)
    if not sequence or sequence[0] != 8:
        return False
    before = sorted(pieces(env))
    for action in (1, 2):
        branch = env.clone()
        branch.step(action)
        if avatar_top(branch) == avatar_top(env):
            continue
        after = sorted(pieces(branch))
        if before != after and row_sequence(branch) and row_sequence(branch)[0] == 8:
            return True
    return False


def macro_search(root, max_states=3000, max_depth=24):
    counter = 0
    start_key = key(root)
    queue = [(-score(root), 0, counter, [])]
    seen = {start_key}
    best = (score(root), [], row_sequence(root))
    run_limits = ((1, 6), (2, 6), (3, 7), (4, 7))
    expanded = 0
    while queue and len(seen) <= max_states:
        _neg_score, depth, _counter, path = heapq.heappop(queue)
        node = p.replay(root, path)
        expanded += 1
        if controlled_eight_prefix(node):
            return path, node, expanded, len(seen), best
        if depth >= max_depth or node.terminal():
            continue
        for action, limit in run_limits:
            branch = node.clone()
            for count in range(1, limit + 1):
                branch.step(action)
                branch_key = key(branch)
                if branch_key in seen:
                    continue
                seen.add(branch_key)
                child_path = path + [action] * count
                child_score = score(branch)
                child_sequence = row_sequence(branch)
                if child_score > best[0]:
                    best = (child_score, child_path, child_sequence)
                    print("BEST", best, "STATE", len(seen))
                if branch.levels_completed > root.levels_completed:
                    return child_path, branch, expanded, len(seen), best
                counter += 1
                heapq.heappush(
                    queue,
                    (-child_score, depth + 1, counter, child_path),
                )
        if expanded % 500 == 0:
            print("PROGRESS", expanded, len(seen), best)
    return None, None, expanded, len(seen), best


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    path, node, expanded, seen, best = macro_search(env)
    print("SEARCH", path, expanded, seen, best)
    if node is not None:
        print("FOUND_LEVEL", node.levels_completed, "SEQ", row_sequence(node))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
