import heapq
import json
import sys
import time
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr

PATH = json.load(open("checkpoint.json"))["final_path"]
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
NAME = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}
SPECS = (
    ("small-cross", 9, ((18, 57), (24, 39))),
    ("outline", 11, ((30, 45), (48, 39), (48, 51))),
    ("large-cross", 8, ((9, 9), (15, 3), (15, 36), (27, 9))),
)


def dot(frame):
    rows, cols = (frame[:63] == 0).nonzero()
    return (int(rows[0]), int(cols[0])) if len(rows) == 1 else None


def bare_frame(root):
    parked = root.clone()
    for _ in range(8):
        parked.step(DOWN)
    return arr(parked.frame())[:63].copy()


def visible_shape(frame, bare, center):
    changed = frame[:63] != bare
    counts = Counter(
        int(frame[row, col])
        for row, col in zip(*changed.nonzero())
        if int(frame[row, col]) not in (0, 1, 2, 4, 5, 15)
    )
    if not counts:
        return None
    color = counts.most_common(1)[0][0]
    points = {
        (int(row), int(col))
        for row, col in zip(*((frame[:63] == color) & changed).nonzero())
    }
    if 0 <= center[0] < 63 and 0 <= center[1] < 64 and frame[center] == 0:
        points.add(center)
    return center, color, points


def descriptor(frame, bare, kind, center):
    shape = visible_shape(frame, bare, center)
    if shape is None:
        return None
    center, color, points = shape
    if kind == "outline":
        rows = [row for row, _ in points]
        cols = [col for _, col in points]
        return color, center, (min(rows), min(cols), max(rows), max(cols)), points
    row_counts = Counter(row for row, _ in points)
    col_counts = Counter(col for _, col in points)
    center_row = row_counts.most_common(1)[0][0]
    center_col = col_counts.most_common(1)[0][0]
    vertical = [row for row, col in points if col == center_col]
    horizontal = [col for row, col in points if row == center_row]
    if not vertical or not horizontal:
        return None
    axes = (min(vertical), center_row, max(vertical),
            min(horizontal), center_col, max(horizontal))
    return color, center, axes, points


def covered(desc, kind, targets):
    if desc is None:
        return ()
    if kind == "outline":
        r0, c0, r1, c1 = desc[2]
        return tuple(
            point for point in targets
            if ((point[0] in (r0, r1) and c0 <= point[1] <= c1)
                or (point[1] in (c0, c1) and r0 <= point[0] <= r1))
        )
    r0, cr, r1, c0, cc, c1 = desc[2]
    return tuple(
        point for point in targets
        if ((point[1] == cc and r0 <= point[0] <= r1)
            or (point[0] == cr and c0 <= point[1] <= c1))
    )


def heuristic(desc, kind, wanted_color, targets):
    if desc is None:
        return 999
    color_penalty = 5 if desc[0] != wanted_color else 0
    hit = len(covered(desc, kind, targets))
    if kind == "outline":
        r0, c0, r1, c1 = desc[2]
        distances = [
            min(abs(r-r0), abs(r-r1), abs(c-c0), abs(c-c1)) // 3
            for r, c in targets
        ]
    else:
        r0, cr, r1, c0, cc, c1 = desc[2]
        distances = [
            min(abs(r-cr), abs(c-cc)) // 3 for r, c in targets
        ]
    return color_penalty + 7 * (len(targets) - hit) + sum(distances)


def search(root, kind, wanted_color, targets, max_states=60000, max_depth=58):
    bare = bare_frame(root)
    frame = arr(root.frame())
    start_center = dot(frame)
    start_desc = descriptor(frame, bare, kind, start_center)
    print(" start", dot(frame), start_desc[:3] if start_desc else None)
    print(" first", [
        (NAME[action], dot(arr((lambda child: (child.step(action), child)[1])(root.clone()).frame())))
        for action in (UP, DOWN, LEFT, RIGHT)
    ])
    queue = [(heuristic(start_desc, kind, wanted_color, targets),
              0, 0, root.clone(), (), start_center)]
    seen = {(frame[:63].tobytes(), start_center): 0}
    serial = 0
    best = (-1, None, ())
    started = time.time()
    while queue and len(seen) < max_states:
        _, _, depth, node, path, center = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            try:
                child.step(action)
                child_frame = arr(child.frame())
            except Exception:
                continue
            dr, dc = {UP: (-3, 0), DOWN: (3, 0),
                      LEFT: (0, -3), RIGHT: (0, 3)}[action]
            child_center = (center[0] + dr, center[1] + dc)
            if not (0 <= child_center[0] < 63 and 0 <= child_center[1] < 64):
                continue
            key = (child_frame[:63].tobytes(), child_center)
            new_depth = depth + 1
            if seen.get(key, 999) <= new_depth:
                continue
            seen[key] = new_depth
            new_path = path + (action,)
            desc = descriptor(child_frame, bare, kind, child_center)
            hit = covered(desc, kind, targets)
            score = (len(hit), desc[0] == wanted_color if desc else False)
            rank = 2 * score[0] + int(score[1])
            if rank > best[0]:
                best = (rank, desc[:3] if desc else None, new_path)
                print(" best", score, best[1], len(new_path),
                      "".join(NAME[a] for a in new_path))
            if desc and desc[0] == wanted_color and len(hit) == len(targets):
                print(" SOLVED", len(new_path), "".join(NAME[a] for a in new_path),
                      desc[:3], "states", len(seen),
                      "secs", round(time.time() - started, 1))
                return new_path
            serial += 1
            h = heuristic(desc, kind, wanted_color, targets)
            heapq.heappush(
                queue, (new_depth + 2 * h, serial, new_depth, child, new_path,
                        child_center)
            )
    print(" FAILED", len(seen), "best", best[0], best[1],
          "".join(NAME[a] for a in best[2]),
          "secs", round(time.time() - started, 1))
    return None


def probe(env):
    for action in PATH:
        env.step(action)
    selected = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for index, (kind, color, targets) in enumerate(SPECS):
        if selected is not None and index != selected:
            continue
        root = env.clone()
        for _ in range(index):
            root.step(USE)
        print("SEARCH", kind, color, targets)
        search(root, kind, color, targets, max_states=15000, max_depth=48)


if __name__ == "__main__":
    A.run_program("re86", probe)
