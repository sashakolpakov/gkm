import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_finish8 import PREFIX, centers


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = env.levels_completed
    ring_mask = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(env.frame()[row][col]) == 9
    }
    targets = ((19, 7), (19, 56), (55, 7), (55, 56))
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def pixels(node):
        frame = node.frame()
        return tuple(
            (row, col)
            for row in range(10, 64)
            for col in range(64)
            if int(frame[row][col]) == 7
        )

    def key(node):
        frame = node.frame()
        square = tuple(
            blob.bbox
            for blob in connected_components(frame, colors=(8,), min_area=9)
            if blob.bbox[0] >= 10
        )
        return square, pixels(node)

    def score(node):
        frame = node.frame()
        overlap = sum(
            point in ring_mask for point in pixels(node)
        )
        body_distance = sum(
            min(max(abs(row - tr), abs(col - tc)) for tr, tc in targets)
            for row, col in centers(frame, 7)
        )
        return overlap, -body_distance

    beam = [(root, [])]
    seen = {key(root)}
    best_overlap = score(root)[0]
    print("root", score(root), centers(root.frame(), 7))
    for depth in range(1, 51):
        candidates = []
        for node, path in beam:
            actions = [(6, col, row) for row, col in pixels(node)]
            actions.append((6, 32, 32))
            for action in actions:
                child = node.clone()
                child.step(*action)
                child_path = path + [action]
                if child.levels_completed > start_level:
                    print("FOUND", PREFIX + child_path)
                    return
                if child.terminal():
                    continue
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                candidates.append((score(child), child, child_path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        bucket_counts = {}
        for item in candidates:
            overlap = item[0][0]
            if bucket_counts.get(overlap, 0) >= 4:
                continue
            bucket_counts[overlap] = bucket_counts.get(overlap, 0) + 1
            selected.append(item)
        beam = [(node, path) for _, node, path in selected[:100]]
        if not beam:
            break
        if candidates[0][0][0] > best_overlap:
            best_overlap = candidates[0][0][0]
            print("NEW_BEST", depth, candidates[0][0],
                  centers(candidates[0][1].frame(), 7),
                  "suffix", candidates[0][2])
        print("depth", depth, "best", candidates[0][0],
              centers(candidates[0][1].frame(), 7), "seen", len(seen))
    print("NO_PATH", len(seen))


A.run_program("su15", inspect)
