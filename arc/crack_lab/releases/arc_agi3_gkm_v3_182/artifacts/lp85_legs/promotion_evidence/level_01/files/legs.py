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
