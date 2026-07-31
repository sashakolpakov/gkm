import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import players
from perception import action_deltas, arr, color_counts, frame_delta, object_candidates, replay


def row_runs(frame):
    grid = arr(frame)
    encoded = []
    for row in grid:
        runs = []
        start = 0
        for col in range(1, len(row) + 1):
            if col == len(row) or row[col] != row[start]:
                runs.append(f"{start}-{col - 1}:{int(row[start])}")
                start = col
        encoded.append(" ".join(runs))
    start = 0
    for row in range(1, len(encoded) + 1):
        if row == len(encoded) or encoded[row] != encoded[start]:
            print(f"R{start}-{row - 1}", encoded[start])
            start = row


def transition_counts(before, after):
    a, b = arr(before), arr(after)
    changed = a != b
    pairs, counts = __import__("numpy").unique(
        __import__("numpy").stack((a[changed], b[changed]), axis=1),
        axis=0,
        return_counts=True,
    )
    return {f"{int(pair[0])}>{int(pair[1])}": int(count) for pair, count in zip(pairs, counts)}


def summary(base, path):
    node = replay(base, path)
    delta = frame_delta(base.frame(), node.frame())
    pieces = [
        (obj["bbox"], obj["area"])
        for obj in object_candidates(node.frame(), min_area=2)
        if obj["color"] == 5
    ]
    print(
        path,
        "reward", node.levels_completed,
        "delta", (delta["count"], delta["bbox"]),
        "trans", transition_counts(base.frame(), node.frame()),
        "five", pieces,
    )


def logical_cells(frame, color):
    grid = arr(frame)
    out = set()
    for row in range(21):
        for col in range(21):
            if int((grid[row * 3:row * 3 + 3, col * 3:col * 3 + 3] == color).sum()) >= 5:
                out.add((row, col))
    return out


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COLORS", color_counts(env.frame()))
    piece = logical_cells(env.frame(), 5)
    targets = logical_cells(env.frame(), 11)
    print("PIECE_CELLS", sorted(piece))
    print("TARGET_CELLS", sorted(targets))
    shape = {(r - min(x for x, _ in piece), c - min(y for _, y in piece)) for r, c in piece}
    matches = []
    for dr in range(21):
        for dc in range(21):
            placed = {(r + dr, c + dc) for r, c in shape}
            overlap = len(placed & targets)
            if overlap:
                matches.append((overlap, dr, dc))
    print("TARGET_MATCHES", sorted(matches, reverse=True)[:8])
    print("OBJECTS")
    for obj in object_candidates(env.frame()):
        if obj["color"] != 4:
            print(obj)
    print("DELTAS")
    for action, delta in action_deltas(env, env.actions).items():
        print(action, {k: delta[k] for k in ("count", "bbox")})
    print("SELECTION_CYCLE")
    for uses in range(8):
        node = replay(env, [5] * uses)
        responses = {}
        for action in (1, 2, 3, 4, 6):
            child = node.clone()
            child.step(action)
            transitions = transition_counts(node.frame(), child.frame())
            responses[action] = {
                pair: count for pair, count in transitions.items()
                if "5" in pair or "10" in pair or "4" in pair
            }
        print(uses, responses)
    print("CONTEXTS")
    for path in (
        [1], [2], [3], [4], [5], [6],
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
        [5, 5, 1], [5, 5, 2],
        [5, 1, 1, 1], [5, 2, 2, 2], [5, 3, 3, 3], [5, 4, 4, 4],
        [5, 5, 3], [5, 5, 4],
        [5, 5] + [1] * 7 + [3] * 10,
        [5, 5] + [1] * 7 + [3] * 10 + [5],
        [5, 5] + [1] * 7 + [3] * 10 + [5] + [2] * 16,
        [5, 5] + [1] * 7 + [3] * 10 + [5] + [1] * 16,
        [5, 5] + [1] * 7 + [3] * 10 + [5] * 60,
        [1] * 5 + [5, 5] + [1] * 7 + [3] * 10 + [5] + [2] * 20,
        [1] * 5 + [5, 5] + [1] * 7 + [3] * 10 + [5] + [2] * 20
        + [5] + [4] * 20,
        [2] * 15 + [5, 5] + [1] * 7 + [3] * 10 + [5] + [1] * 20,
        [5, 5] + [2] * 4,
        [5, 5] + [2] * 5,
        [5, 5] + [4] * 2,
        [5, 5] + [4] * 3,
        [5, 5] + [2] * 4 + [4] * 2,
    ):
        summary(env, path)


levels, path, err = A.run_program("ar25", probe)
print("PROBE_RESULT", levels, len(path), err)
