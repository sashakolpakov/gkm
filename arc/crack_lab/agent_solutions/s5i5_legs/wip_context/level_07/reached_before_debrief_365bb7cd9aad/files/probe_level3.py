import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr
from players import play_level_1, play_level_2


def probe(env):
    play_level_1(env)
    play_level_2(env)
    print("level", env.levels_completed, "actions", env.actions)
    def markers(node):
        grid = arr(node.frame())
        targets = []
        tips = []
        for r in range(1, 63):
            for c in range(1, 63):
                if all(
                    int(grid[rr, cc]) == 13
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    targets.append((r, c))
                if int(grid[r, c]) == 13:
                    tips.append((r, c))
        return tips, targets

    controls = {
        "a_down": (10, 48),
        "a_up": (16, 48),
        "tip10_left": (29, 48),
        "tip10_right": (35, 48),
        "c_right": (48, 48),
        "c_left": (54, 48),
        "blue_right": (10, 57),
        "blue_left": (16, 57),
        "tip7_up": (29, 57),
        "tip7_down": (35, 57),
        "yellow_down": (48, 57),
        "yellow_up": (54, 57),
    }
    def distance(node):
        tips, targets = markers(node)
        outline = {
            point
            for r, c in targets
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        }
        tips = [point for point in tips if point not in outline]
        return min(
            sum(
                abs(tip[0] - target[0]) + abs(tip[1] - target[1])
                for tip, target in zip(tips, order)
            )
            for order in (targets, targets[::-1])
        )

    actions = [(name, (6, *point)) for name, point in controls.items()]
    serial = 0
    queue = [(distance(env), 0, serial, [])]
    seen = {arr(env.frame())[:40].tobytes()}
    best = distance(env)
    while queue and len(seen) < 10000:
        _, _, _, path = heapq.heappop(queue)
        node = env.clone()
        for name in path:
            node.step(6, *controls[name])
        if len(seen) % 500 < 12:
            print("search", len(seen), "queue", len(queue), "depth", len(path), "h", distance(node))
        if len(path) >= 72:
            continue
        for name, action in actions:
            child = node.clone()
            child.step(*action)
            key = arr(child.frame())[:40].tobytes()
            if key in seen:
                continue
            child_path = path + [name]
            if child.levels_completed > 2:
                print("WIN", child_path)
                return
            seen.add(key)
            serial += 1
            h = distance(child)
            if h < best:
                best = h
                print("BEST", best, child_path, "tips", markers(child)[0])
            heapq.heappush(queue, (h + 0.03 * len(child_path), h, serial, child_path))
    print("no win", len(seen), "queue", len(queue))


arena.run_program("s5i5", probe)
