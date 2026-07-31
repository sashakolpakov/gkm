# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

import numpy as np
from collections import deque


def play_fixed_sequence(env, sequence):
    """Play a fixed list of actions on the real env."""
    for act in sequence:
        if env.terminal():
            break
        env.step(int(act))


def click_select(env, x, y):
    """Coordinate interaction: issue action 6 at pixel (x=col, y=row).
    In sp80 this selects/grabs the object under the cursor so that the
    directional actions (1..4) then move THAT object.  Recorded in replay
    paths as [6, x, y]."""
    env.step(6, int(x), int(y))


def grab_and_move(env, x, y, moves):
    """Select-then-drive: grab the object under pixel (x=col, y=row) with the
    coordinate action (action 6), then play a fixed list of directional
    *moves* (1..4) on that now-selected object.

    This is the recurring skill for sp80's multi-object levels: any block can
    be driven by first grabbing it, then issuing directional steps. Composes
    `click_select` + `play_fixed_sequence` so the pattern is written ONCE."""
    click_select(env, x, y)
    play_fixed_sequence(env, moves)


def drive_objects(env, plan, commit=(5,)):
    """Multi-object drive: the recurring sp80 select-then-drive-then-commit
    idiom for levels with several independently-movable blocks.

    `plan` is a declarative list of `((x, y), moves)` tuples: WHERE to grab
    each object (pixel col=x, row=y) and HOW to move it (a list of directional
    actions 1..4). Each tuple is `grab_and_move`'d in turn, then the `commit`
    sequence (default USE=5) is played once to lock in / win the configuration.

    This is the higher-order leg factored out of `play_level_2` and
    `play_level_3`, which were both just: N grab_and_move calls + final USE.
    A player collapses to a single declarative `plan`; all iteration and commit
    logic lives here, written ONCE. Emits exactly the same action stream as the
    old inline code (no behaviour change)."""
    for (x, y), moves in plan:
        grab_and_move(env, x, y, moves)
    play_fixed_sequence(env, list(commit))


def probe_level_objects(env):
    """Print compact clone-only observations for an unsolved object level."""
    from perception import action_deltas, connected_components, frame_delta

    base = np.asarray(env.frame())
    blobs = connected_components(base, min_area=4)
    print("L5_COUNTS", {
        int(color): int((base == color).sum())
        for color in np.unique(base)
    })
    print("L5_BLOBS", [
        (b.color, b.bbox, b.area) for b in blobs
        if b.color != 12
    ])
    for b in blobs:
        if b.color not in (8, 9, 11, 15):
            continue
        r0, c0, r1, c1 = b.bbox
        mask = base[r0:r1 + 1, c0:c1 + 1] == b.color
        print("L5_MASK", b.color, b.bbox,
              tuple("".join("#" if value else "." for value in row)
                    for row in mask))
    deltas = action_deltas(env, actions=(1, 2, 3, 4, 5))
    print("L5_KEYS", {
        action: (delta["count"], delta["bbox"])
        for action, delta in deltas.items()
    })
    for b in blobs:
        if b.color in (1, 12) or b.area > 300:
            continue
        r0, c0, r1, c1 = b.bbox
        x, y = (c0 + c1) // 2, (r0 + r1) // 2
        picked = env.clone()
        picked.step(6, x, y)
        select_delta = frame_delta(base, picked.frame())
        moves = {}
        for action in (1, 2, 3, 4):
            moved = picked.clone()
            before = np.asarray(picked.frame()).copy()
            moved.step(action)
            delta = frame_delta(before, moved.frame())
            moves[action] = (delta["count"], delta["bbox"])
        print("L5_CLICK", (b.color, b.bbox, b.area), (x, y),
              (select_delta["count"], select_delta["bbox"]), moves)


def probe_known_local(env, label, pieces, winning, step):
    """Test one-step positional sensitivity around a known winning layout."""
    from perception import connected_components

    blobs = connected_components(env.frame(), colors=(8, 9, 11, 15),
                                 min_area=4)
    print(label + "_GEOM", [
        (b.color, b.bbox, b.area) for b in blobs
    ])
    outcomes = []
    for varied in range(len(pieces)):
        row = []
        for offset in (-2, -1, 0, 1, 2):
            targets = list(winning)
            targets[varied] += offset * step
            node = env.clone()
            for (x, y, start), target in zip(pieces, targets):
                node.step(6, x, y)
                action = 3 if target < start else 4
                for _ in range(abs(target - start) // step):
                    node.step(action)
            node.step(5)
            row.append(int(node.levels_completed > env.levels_completed))
        outcomes.append(tuple(row))
    print(label + "_LOCAL", tuple(outcomes))


def probe_level5_candidates(env):
    """Commit a few geometry-derived level-5 bridge candidates on clones."""
    candidates = (
        ("start", ()),
        ("D_up1", (((34, 43), (1,)),)),
        ("D_up2", (((34, 43), (1, 1)),)),
        ("D_up3", (((34, 43), (1, 1, 1)),)),
        ("B_down1", (((24, 33), (2,)),)),
        ("B_left1", (((24, 33), (3,)),)),
        ("B_left1_down1", (((24, 33), (3, 2)),)),
        ("B_left1_C_down1", (((24, 33), (3,)),
                             ((48, 33), (2,)))),
        ("B_left1_A_down5", (((24, 33), (3,)),
                             ((34, 21), (2, 2, 2, 2, 2)))),
        ("B_left1_D_up2", (((24, 33), (3,)),
                           ((34, 43), (1, 1)))),
        ("x_chain", (((24, 33), (3,)),
                     ((34, 21), (3,)),
                     ((34, 43), (4, 4)))),
        ("x_chain_D_up1", (((24, 33), (3,)),
                           ((34, 21), (3,)),
                           ((34, 43), (4, 4, 1)))),
        ("x_chain_C_down1_D_up1", (
            ((24, 33), (3,)),
            ((34, 21), (3,)),
            ((48, 33), (2,)),
            ((34, 43), (4, 4, 1)),
        )),
        ("x_chain_A_down5_D_up1", (
            ((24, 33), (3,)),
            ((34, 21), (3,) + (2,) * 5),
            ((34, 43), (4, 4, 1)),
        )),
        ("x_chain_B_down1_D_up1", (
            ((24, 33), (3, 2)),
            ((34, 21), (3,)),
            ((34, 43), (4, 4, 1)),
        )),
        ("C_down1", (((48, 33), (2,)),)),
        ("B_right", (((24, 33), (4,)),)),
        ("A_left", (((34, 21), (3,)),)),
        ("A_left2", (((34, 21), (3, 3)),)),
        ("B_right_D_up", (((24, 33), (4,)), ((34, 43), (1,)))),
        ("B_right_D_up2", (((24, 33), (4,)),
                           ((34, 43), (1, 1)))),
        ("B_right_D_up3", (((24, 33), (4,)),
                           ((34, 43), (1, 1, 1)))),
        ("A_left_D_up", (((34, 21), (3,)), ((34, 43), (1,)))),
        ("A_left_D_up2", (((34, 21), (3,)),
                          ((34, 43), (1, 1)))),
        ("A_left2_D_up2_B_down3", (
            ((34, 21), (3, 3)),
            ((34, 43), (1, 1)),
            ((24, 33), (2, 2, 2)),
        )),
        ("A_left2_D_up1_B_down1", (
            ((34, 21), (3, 3)),
            ((34, 43), (1,)),
            ((24, 33), (2,)),
        )),
        ("A_left2_vertical_stair", (
            ((34, 21), (3, 3) + (2,) * 7),
            ((24, 33), (2,)),
            ((48, 33), (2, 2)),
        )),
        ("both_D_up1", (((24, 33), (4,)), ((34, 21), (3,)),
                        ((34, 43), (1,)))),
        ("both_D_up2", (((24, 33), (4,)), ((34, 21), (3,)),
                        ((34, 43), (1, 1)))),
        ("B_right_B_down_D_up2", (((24, 33), (4, 2)),
                                  ((34, 43), (1, 1)))),
        ("A_left_C_down_D_up2", (((34, 21), (3,)),
                                 ((48, 33), (2,)),
                                 ((34, 43), (1, 1)))),
        ("B_right_C_down", (((24, 33), (4,)), ((48, 33), (2,)))),
    )
    outcomes = []
    for name, plan in candidates:
        node = env.clone()
        for (x, y), moves in plan:
            node.step(6, x, y)
            for action in moves:
                node.step(action)
        node.step(5)
        outcomes.append((name, int(node.levels_completed >
                                   env.levels_completed)))
    print("L5_CANDIDATES", tuple(outcomes))


def search_level5_dense(env, trials=800):
    """Search clone states whose x/y projections cover both target spans."""
    import itertools
    import random

    def covers(intervals, lo, hi):
        intervals = sorted(intervals)
        end = lo - 1
        for left, right in intervals:
            if right < lo or left > end + 1:
                continue
            end = max(end, right)
            if end >= hi:
                return True
        return False

    x_values = (
        tuple(range(5, 51, 3)),
        tuple(range(14, 54, 3)),
        tuple(range(14, 48, 3)),
        tuple(range(14, 57, 3)),
    )
    y_values = (
        tuple(range(14, 51, 3)),
        tuple(range(14, 51, 3)),
        tuple(range(14, 51, 3)),
        tuple(range(14, 48, 3)),
    )
    widths = (12, 9, 15, 6)
    heights = (3, 3, 3, 6)
    xs = [
        values for values in itertools.product(*x_values)
        if covers([(left, left + width - 1)
                   for left, width in zip(values, widths)], 17, 55)
    ]
    ys = [
        values for values in itertools.product(*y_values)
        if covers([(top, top + height - 1)
                   for top, height in zip(values, heights)], 35, 43)
    ]
    print("L5_DENSE_SPACE", len(xs), len(ys))

    # C, B, A, D: moving C first avoids the initially-selected C being
    # mistaken for another piece. Reject layouts that cover a later grab.
    order = (2, 1, 0, 3)
    centers = ((34, 21), (24, 33), (48, 33), (34, 43))
    starts_x = (29, 20, 41, 32)
    starts_y = (20, 32, 32, 41)
    rng = random.Random(5805)
    checked = 0
    for _ in range(trials * 3):
        if checked >= trials:
            break
        tx = rng.choice(xs)
        ty = rng.choice(ys)
        clear = True
        for pos, piece in enumerate(order):
            left, top = tx[piece], ty[piece]
            for later in order[pos + 1:]:
                cx, cy = centers[later]
                if (left <= cx < left + widths[piece] and
                        top <= cy < top + heights[piece]):
                    clear = False
                    break
            if not clear:
                break
        if not clear:
            continue
        checked += 1
        node = env.clone()
        plan = []
        for piece in order:
            x, y = centers[piece]
            moves = []
            dx = tx[piece] - starts_x[piece]
            dy = ty[piece] - starts_y[piece]
            moves.extend(([3] if dx < 0 else [4]) * (abs(dx) // 3))
            moves.extend(([1] if dy < 0 else [2]) * (abs(dy) // 3))
            node.step(6, x, y)
            for action in moves:
                node.step(action)
            plan.append(((x, y), tuple(moves)))
        node.step(5)
        if node.levels_completed > env.levels_completed:
            print("L5_DENSE_WIN", checked, tuple(plan), tx, ty)
            return
    print("L5_DENSE_WIN", None, checked)


def make_bbox_key(color):
    """Return a compact key function that tracks the bounding box of pixels
    with the given color value.  Returns None if the color is absent (BFS
    treats that as an already-seen / skip state)."""
    def key_fn(env):
        f = np.array(env.frame())
        rows, cols = np.where(f == color)
        if len(rows) == 0:
            return None
        return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
    return key_fn


def bfs_or_fallback(env, key_fn, fallback,
                    actions=(1, 2, 3, 4, 5, 6),
                    max_states=500, max_depth=20):
    """Try BFS (compact key) to find a winning path; play it.
    If BFS finds nothing, play *fallback* (a fixed action list) instead."""
    path = bfs_win_compact(env, key_fn, actions=actions,
                           max_states=max_states, max_depth=max_depth)
    play_fixed_sequence(env, path if path else fallback)


def bfs_win(env, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):
    """BFS over clone states to find shortest action sequence that increases
    env.levels_completed.  Returns the path (list of ints) or None."""
    base_level = env.levels_completed

    def key_fn(e):
        f = np.array(e.frame())
        return f.tobytes()

    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}

    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            child.step(int(action))
            if child.levels_completed > base_level:
                return path + [int(action)]
            k = key_fn(child)
            if k in seen:
                continue
            seen.add(k)
            q.append((child, path + [int(action)]))
    return None


def bfs_win_compact(env, key_fn, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):
    """BFS with a custom key function (cheaper than full frame hash)."""
    base_level = env.levels_completed
    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}

    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            child.step(int(action))
            if child.levels_completed > base_level:
                return path + [int(action)]
            k = key_fn(child)
            if k is None or k in seen:
                continue
            seen.add(k)
            q.append((child, path + [int(action)]))
    return None
