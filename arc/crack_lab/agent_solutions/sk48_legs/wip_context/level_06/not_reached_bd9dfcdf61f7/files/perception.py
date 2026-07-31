"""Source-free frame perception helpers for cracking.

This module is deliberately observational: it derives compact symbolic state
from `env.frame()` and `env.clone()` only. It is a cofibration-style scaffold:
raw pixels are embedded into a monotone tower of reusable observations
(components -> objects -> action deltas -> replay states). Candidate level
logic should be written against these quotients, then replay-validated by the
harness. No game source or prior solution history is read here.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
ACTIONS = (UP, DOWN, LEFT, RIGHT, USE)
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAME = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT", USE: "USE"}


@dataclass(frozen=True)
class Blob:
    color: int
    bbox: Tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    area: int
    centroid: Tuple[float, float]

    @property
    def top_left(self):
        return self.bbox[0], self.bbox[1]

    @property
    def size(self):
        r0, c0, r1, c1 = self.bbox
        return r1 - r0 + 1, c1 - c0 + 1


def arr(frame) -> np.ndarray:
    return np.asarray(frame)


def color_counts(frame) -> Dict[int, int]:
    vals, cnts = np.unique(arr(frame), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def connected_components(frame, colors: Optional[Iterable[int]] = None,
                         min_area: int = 1) -> List[Blob]:
    f = arr(frame)
    wanted = None if colors is None else {int(c) for c in colors}
    seen = np.zeros(f.shape, dtype=bool)
    out: List[Blob] = []
    rows, cols = f.shape[:2]
    for r in range(rows):
        for c in range(cols):
            if seen[r, c]:
                continue
            color = int(f[r, c])
            if wanted is not None and color not in wanted:
                seen[r, c] = True
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not seen[nx, ny] and int(f[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if len(pts) >= min_area:
                rs = [p[0] for p in pts]
                cs = [p[1] for p in pts]
                out.append(Blob(color, (min(rs), min(cs), max(rs), max(cs)),
                                len(pts), (sum(rs) / len(pts), sum(cs) / len(pts))))
    return sorted(out, key=lambda b: (b.color, b.bbox))


def block_signatures(frame, cell: int = 4) -> Dict[Tuple[int, int], Tuple[int, ...]]:
    """Partition a frame into fixed cells and return each cell's color signature."""
    f = arr(frame)
    out = {}
    for r in range(0, f.shape[0], cell):
        for c in range(0, f.shape[1], cell):
            out[(r // cell, c // cell)] = tuple(int(v) for v in sorted(np.unique(f[r:r+cell, c:c+cell])))
    return out


def object_candidates(frame, cell: int = 4, min_area: int = 4) -> List[dict]:
    """A compact, game-agnostic object list from color components and cell signatures."""
    f = arr(frame)
    blobs = connected_components(f, min_area=min_area)
    sigs = block_signatures(f, cell)
    objects = []
    for b in blobs:
        r0, c0, r1, c1 = b.bbox
        objects.append({
            "color": b.color,
            "bbox": b.bbox,
            "top_left": b.top_left,
            "size": b.size,
            "area": b.area,
            "centroid": b.centroid,
            "cell": (r0 // cell, c0 // cell),
            "cell_sig": sigs.get((r0 // cell, c0 // cell)),
        })
    return objects


def frame_delta(before, after) -> dict:
    a, b = arr(before), arr(after)
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return {"count": 0, "bbox": None, "samples": []}
    samples = [(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys[:80], xs[:80])]
    return {
        "count": int(len(ys)),
        "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        "samples": samples,
    }


def action_deltas(env, actions: Sequence[int] = ACTIONS) -> Dict[int, dict]:
    base = arr(env.frame()).copy()
    out = {}
    for action in actions:
        clone = env.clone()
        clone.step(action)
        out[int(action)] = frame_delta(base, clone.frame())
    return out


def replay(env, actions: Sequence[int]):
    clone = env.clone()
    for action in actions:
        if clone.terminal():
            break
        clone.step(int(action))
    return clone


def path_result(env, actions: Sequence[int]) -> dict:
    clone = replay(env, actions)
    return {
        "levels_completed": int(clone.levels_completed),
        "terminal": bool(clone.terminal()),
        "path_len": len(actions),
        "colors": color_counts(clone.frame()),
        "objects": object_candidates(clone.frame()),
    }


def changed_signature(env, actions: Sequence[int], cell: int = 4):
    before = block_signatures(env.frame(), cell)
    clone = replay(env, actions)
    after = block_signatures(clone.frame(), cell)
    return {k: (before.get(k), after.get(k)) for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)}


def bounded_bfs(env, goal_fn, actions: Sequence[int] = (UP, DOWN, LEFT, RIGHT, USE),
                key_fn=None, max_states: int = 20000, max_depth: int = 80):
    """Generic clone BFS over observational keys. Use small max_states first."""
    if key_fn is None:
        key_fn = lambda e: arr(e.frame()).tobytes()
    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}
    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if goal_fn(node, path):
            return path
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            child.step(int(action))
            key = key_fn(child)
            if key in seen:
                continue
            seen.add(key)
            q.append((child, path + [int(action)]))
    return None


def bounded_replay_bfs(env, goal_fn, action_fn,
                       key_fn=None, max_states: int = 20000, max_depth: int = 80):
    """Path-only BFS for games whose deep Arena clones become expensive.

    The queue retains compact action paths, not recursively deep-copied runtime
    states. Each node is reconstructed from one root clone. ``action_fn(node)``
    may return integer actions or coordinate tuples such as ``(6, x, y)``.
    """
    if key_fn is None:
        key_fn = lambda e: arr(e.frame()).tobytes()

    def reconstruct(path):
        node = env.clone()
        for action in path:
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(int(action))
        return node

    start = reconstruct([])
    q = deque([[]])
    seen = {key_fn(start)}
    while q and len(seen) <= max_states:
        path = q.popleft()
        node = reconstruct(path)
        if goal_fn(node, path):
            return path
        if len(path) >= max_depth or node.terminal():
            continue
        for action in action_fn(node):
            child_path = path + [action]
            child = reconstruct(child_path)
            key = key_fn(child)
            if key in seen:
                continue
            seen.add(key)
            if goal_fn(child, child_path):
                return child_path
            q.append(child_path)
    return None


def level_goal(base_level: int):
    return lambda env, path: env.levels_completed > base_level
