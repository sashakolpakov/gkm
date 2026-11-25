"""Reusable skills inferred from public frames and verified clone actions."""
import json
import math
from collections import Counter
from perception import arr, connected_components, safe_step


def align_ring_marker(env):
    """Rotate a colored perimeter until its block matches the corner marker.

    Read square blocks, the four small marker corners, and the two arrow
    components from the frame. Learn each arrow's rotation on a clone, then
    minimize the circular block distance to the marker. Commit only a path
    that has already increased the level reward on a clone.
    """
    frame = arr(env.frame())
    blobs = connected_components(frame)
    blocks = [b for b in blobs if b.area == 16 and b.size == (4, 4)]
    corners = [b for b in blobs if b.area == 4 and b.size == (2, 2)]
    if not blocks or len(corners) != 4:
        raise RuntimeError('Expected square ring blocks and four marker corners')
    color = Counter(b.color for b in corners).most_common(1)[0][0]
    marker = tuple(sum(b.centroid[i] for b in corners) / 4 for i in (0, 1))
    cy = sum(b.centroid[0] for b in blocks) / len(blocks)
    cx = sum(b.centroid[1] for b in blocks) / len(blocks)
    blocks.sort(key=lambda b: math.atan2(b.centroid[0] - cy, b.centroid[1] - cx))
    slots = [(int(b.centroid[1]), int(b.centroid[0])) for b in blocks]
    goal = min(range(len(blocks)), key=lambda i: sum(
        (blocks[i].centroid[j] - marker[j]) ** 2 for j in (0, 1)))
    arrows = [b for b in blobs if b.area == 40 and b.size == (8, 6)]
    actions = [(6, int(b.centroid[1]), int(b.centroid[0])) for b in arrows]
    if len(actions) != 2:
        raise RuntimeError('Expected two rotation arrows')

    def sequence(e):
        f = arr(e.frame())
        return tuple(int(f[y, x]) for x, y in slots)

    def distance(seq):
        n = len(seq)
        return min(min((i-goal) % n, (goal-i) % n)
                   for i, value in enumerate(seq) if value == color)

    base = env.levels_completed
    clone = env.clone()
    path = []
    for _ in range(len(slots)):
        if clone.levels_completed > base:
            break
        before = sequence(clone)
        candidates = []
        for action in actions:
            trial = clone.clone()
            safe_step(trial, action)
            if trial.levels_completed > base:
                candidates.append((-1, action, trial))
                continue
            after = sequence(trial)
            if after not in (before[1:] + before[:1], before[-1:] + before[:-1]):
                raise RuntimeError('Arrow did not rotate the observed ring')
            if not trial.terminal():
                candidates.append((distance(after), action, trial))
        if not candidates:
            raise RuntimeError('No viable ring rotation')
        score, action, trial = min(candidates, key=lambda item: item[0])
        if score >= distance(before):
            raise RuntimeError('No rotation reduces marker distance')
        path.append(action)
        clone = trial
    if clone.levels_completed <= base:
        raise RuntimeError('Ring alignment did not complete the level')
    with open('ring_alignment_candidate.json', 'w') as out:
        json.dump(path, out)
    for action in path:
        safe_step(env, action)


def align_intersecting_tracks(env):
    """Align marked tiles by rotating intersecting rows and a perimeter.

    Infer tracks from tile and arrow geometry, verify their permutations on
    clones, and search the positions of the requested color. Validate the
    complete path against frames and reward before committing it.
    """
    from collections import deque

    frame = arr(env.frame())
    blobs = connected_components(frame)
    tiles = [b for b in blobs if b.area == 4 and b.size == (2, 2)]
    points = sorted((b.bbox[1], b.bbox[0]) for b in tiles)
    index = {p: i for i, p in enumerate(points)}
    markers = []
    for x, y in points:
        if x < 1 or y < 1 or x + 2 >= 64 or y + 2 >= 64:
            continue
        corners = [int(frame[yy, xx]) for yy in (y-1, y+2)
                   for xx in (x-1, x+2)]
        if len(set(corners)) == 1 and corners[0] != int(frame[y-1, x]):
            markers.append((index[x, y], corners[0]))
    if not markers or len({c for _, c in markers}) != 1:
        raise RuntimeError('Expected markers requesting one tile color')
    color = markers[0][1]
    goal = tuple(sorted(i for i, _ in markers))
    arrows = [b for b in blobs if b.area == 10 and b.size == (4, 3)]
    rows = sorted({y for _, y in points})
    max_width = max(sum(yy == y for _, yy in points) for y in rows)
    actions, permutations = [], []

    def sequence(e):
        f = arr(e.frame())
        return tuple(int(f[y, x]) for x, y in points)

    initial = sequence(env)
    for arrow in arrows:
        y = min(rows, key=lambda yy: abs(yy + .5 - arrow.centroid[0]))
        row = sorted(p for p in points if p[1] == y)
        if len(row) == max_width:
            loop = row
        else:
            left, right = row[0][0], row[-1][0]
            top, bottom = min(rows), max(rows)
            loop = (sorted(p for p in points if p[1] == top and left <= p[0] <= right)
                    + sorted((p for p in points if p[0] == right and top < p[1] <= bottom), key=lambda p: p[1])
                    + sorted((p for p in points if p[1] == bottom and left <= p[0] < right), reverse=True)
                    + sorted((p for p in points if p[0] == left and top < p[1] < bottom), key=lambda p: -p[1]))
        action = (6, int(arrow.centroid[1]), int(arrow.centroid[0]))
        trial = env.clone()
        safe_step(trial, action)
        observed = sequence(trial)
        fits = []
        for shift in (-1, 1):
            perm = list(range(len(points)))
            for j, p in enumerate(loop):
                perm[index[p]] = index[loop[(j + shift) % len(loop)]]
            predicted = list(initial)
            for src, dst in enumerate(perm):
                predicted[dst] = initial[src]
            if tuple(predicted) == observed:
                fits.append(tuple(perm))
        if len(fits) != 1:
            raise RuntimeError('Track rotation was not uniquely verified')
        actions.append(action)
        permutations.append(fits[0])

    start = tuple(i for i, c in enumerate(initial) if c == color)
    queue = deque([(start, [])])
    seen = {start}
    path = None
    while queue and len(seen) <= 10000:
        state, prefix = queue.popleft()
        if state == goal:
            path = prefix
            break
        for k, perm in enumerate(permutations):
            after = tuple(sorted(perm[i] for i in state))
            if after not in seen:
                seen.add(after)
                queue.append((after, prefix + [k]))
    if path is None:
        raise RuntimeError('No bounded track alignment path')
    clone = env.clone()
    base = env.levels_completed
    state = initial
    candidate = []
    for k in path:
        safe_step(clone, actions[k])
        candidate.append(actions[k])
        if clone.levels_completed > base:
            break
        predicted = list(state)
        for src, dst in enumerate(permutations[k]):
            predicted[dst] = state[src]
        state = tuple(predicted)
        if sequence(clone) != state:
            raise RuntimeError('Track model failed in a new context')
    if clone.levels_completed <= base:
        raise RuntimeError('Marked track alignment did not complete level')
    with open('intersecting_tracks_candidate.json', 'w') as out:
        json.dump(candidate, out)
    for action in candidate:
        safe_step(env, action)
