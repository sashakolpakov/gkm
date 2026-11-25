"""Reusable skills inferred from public frame and clone observations."""
from perception import arr, replay, safe_step


def mate_blue_tips(env):
    """Rotate and translate a movable piece until its blue tips meet fixed tips.

    Discover movable blue pixels with an upward clone probe. Use tip alignment
    distance as progress, then verify each complete candidate against reward.
    """
    base_level = env.levels_completed
    candidates = []
    for turns in range(4):
        rotated = replay(env, [5] * turns)
        f = arr(rotated.frame())
        moved = replay(rotated, [1])
        g = arr(moved.frame())
        blue = set(zip(*((f == 8).nonzero())))
        stationary = blue & set(zip(*((g == 8).nonzero())))
        mobile = blue - stationary
        if not mobile or len(mobile) != len(stationary):
            continue
        anchor = min(mobile)
        for target in sorted(stationary):
            dy, dx = int(target[0] - anchor[0]), int(target[1] - anchor[1])
            if dy % 3 or dx % 3:
                continue
            if {(y + dy, x + dx) for y, x in mobile} != stationary:
                continue
            path = ([5] * turns + [2 if dy > 0 else 1] * (abs(dy) // 3)
                    + [4 if dx > 0 else 3] * (abs(dx) // 3))
            candidates.append(path)
    for path in sorted(candidates, key=len):
        trial = replay(env, path)
        if trial.levels_completed <= base_level:
            continue
        for action in path:
            if env.terminal() or env.levels_completed > base_level:
                break
            safe_step(env, action)
        return
    raise RuntimeError('No verified blue-tip alignment path')


def assemble_blue_tips(env):
    """Select rigid pieces and pair every blue tip without overlapping bodies.

    Probe each piece's four orientations. Search assemblies by adding a piece
    at an unpaired tip; the number of paired tips is the dense progress measure.
    Verify a complete assembly on a clone before committing its moves.
    """
    from perception import connected_components, color_counts

    base_level = env.levels_completed
    background = max(color_counts(env.frame()), key=color_counts(env.frame()).get)
    pieces = [b for b in connected_components(env.frame(), min_area=9)
              if b.color not in (background, 8) and b.bbox[0] > 0]
    entry = arr(env.frame())

    def moving_color(probe):
        f = arr(probe.frame())
        g = arr(replay(probe, [1]).frame())
        counts = {}
        for y, x in zip(*((f != g).nonzero())):
            color = int(f[y, x])
            if y > 1 and color not in (background, 8):
                counts[color] = counts.get(color, 0) + 1
        return max(counts, key=counts.get)

    selected_color = moving_color(env)
    pieces.sort(key=lambda b: (b.color != selected_color, b.bbox))
    selections = []
    for b in pieces:
        y, x = next((y, x) for y in range(b.bbox[0], b.bbox[2] + 1)
                    for x in range(b.bbox[1], b.bbox[3] + 1)
                    if entry[y, x] == b.color)
        selections.append(None if b.color == selected_color else (6, int(x), int(y)))
    models = []
    for selection in selections:
        probe = replay(env, [] if selection is None else [selection])
        active_color = moving_color(probe)
        variants = []
        for rotation in range(4):
            f = arr(probe.frame())
            body = {(y, x) for y in range(2, 64, 3) for x in range(2, 64, 3)
                    if f[y, x] == active_color}
            tips = {(y, x) for y in range(2, 64, 3) for x in range(2, 64, 3)
                    if f[y, x] == 8 and any(
                        (y + dy, x + dx) in body
                        for dy, dx in ((3, 0), (-3, 0), (0, 3), (0, -3)))}
            # A change in body size means USE extends or retracts this piece.
            # Keep its entry shape; the marked-piece leg chooses its length.
            if variants and len(body) != len(variants[0][0]):
                variants = variants[:1]
                break
            variants.append((body, tips))
            probe = replay(probe, [5])
        models.append(variants)

    root = max(range(len(models)), key=lambda i: len(models[i][0][0]))
    body, tips = models[root][0]
    stack = [({root: (0, 0, 0)}, body, {p: 1 for p in tips})]
    solutions = []
    visited = 0
    while stack and visited < 50000:
        poses, body, tips = stack.pop()
        visited += 1
        if len(poses) == len(models):
            if all(n == 2 for n in tips.values()):
                solutions.append(poses)
            continue
        for i, variants in enumerate(models):
            if i in poses:
                continue
            seen = set()
            for rotation, (piece_body, piece_tips) in enumerate(variants):
                for py, px in (p for p, n in tips.items() if n == 1):
                    for ty, tx in piece_tips:
                        dy, dx = py - ty, px - tx
                        pose = (rotation, dy, dx)
                        if pose in seen:
                            continue
                        seen.add(pose)
                        nb = {(y + dy, x + dx) for y, x in piece_body}
                        nt = {(y + dy, x + dx) for y, x in piece_tips}
                        if (nb & (body | tips.keys()) or nt & body
                                or any(tips.get(p, 0) == 2 for p in nt)):
                            continue
                        points = body | nb | tips.keys() | nt
                        if any(max(p[k] for p in points) - min(p[k] for p in points) > 60
                               for k in (0, 1)):
                            continue
                        counts = tips.copy()
                        for p in nt:
                            counts[p] = counts.get(p, 0) + 1
                        stack.append((poses | {i: pose}, body | nb, counts))

    for poses in solutions:
        points = set()
        for i, (rotation, dy, dx) in poses.items():
            body, tips = models[i][rotation]
            points |= {(y + dy, x + dx) for y, x in body | tips}
        shifts = [((64 - max(p[k] for p in points) - min(p[k] for p in points)) // 6) * 3
                  for k in (0, 1)]
        path = []
        for i, selection in enumerate(selections):
            rotation, dy, dx = poses[i]
            dy += shifts[0]
            dx += shifts[1]
            path += ([] if selection is None else [selection]) + [5] * rotation
            path += [2 if dy > 0 else 1] * (abs(dy) // 3)
            path += [4 if dx > 0 else 3] * (abs(dx) // 3)
        trial = replay(env, path)
        if trial.levels_completed > base_level:
            for action in path:
                if env.terminal() or env.levels_completed > base_level:
                    break
                safe_step(env, action)
            return
    raise RuntimeError('No verified assembly pairing all blue tips')


def extend_marked_piece(env):
    """Extend a selected telescoping piece until its black frontier disappears.

    Validate each extension by increased body area and reduced remaining frontier.
    Stop at the first fully exposed shape, before USE starts retracting it.
    """
    from perception import color_counts
    for _ in range(20):
        before = color_counts(env.frame())
        if not (arr(env.frame())[2:] == 0).any():
            return
        trial = replay(env, [5])
        after = color_counts(trial.frame())
        if after.get(8, 0) <= before.get(8, 0):
            raise RuntimeError('Marked piece did not extend')
        safe_step(env, 5)
    raise RuntimeError('Marked piece extension bound exceeded')
