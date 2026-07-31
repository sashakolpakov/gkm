import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def components(env):
    return P.connected_components(env.frame(), colors=(1, 9, 10, 11, 12, 14), min_area=2)


def key(env):
    return tuple((b.color, b.bbox, b.area) for b in components(env))


def score(env):
    mains = [b for b in components(env) if b.area == 16 and b.color in (1, 10)]
    if len(mains) != 2:
        return 10000
    (a, b) = sorted(mains, key=lambda item: item.bbox[1])
    ar, ac = a.bbox[:2]
    br, bc = b.bbox[:2]
    return min(ar, br) * 2 - abs(ar - br) * 8 - abs(abs(ac - bc) - 8) * 3


def apply_macro(env, macro):
    if macro[0] == "click":
        env.step(6, macro[1], macro[2])
        return
    _, direction, count = macro
    for _ in range(count):
        env.step(direction)
        if env.terminal() or env.levels_completed > 5:
            return


def reconstruct(root, path):
    node = root.clone()
    for macro in path:
        apply_macro(node, macro)
        if node.terminal() or node.levels_completed > 5:
            break
    return node


def choices(env):
    result = []
    for blob in components(env):
        if blob.color not in (1, 9):
            continue
        r0, c0, r1, c1 = blob.bbox
        result.append(("click", (c0 + c1) // 2, (r0 + r1) // 2))
    result.extend(
        ("move", direction, count)
        for direction in (1, 2, 3, 4)
        for count in range(1, 9)
    )
    return result


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    start_key = key(root)
    queue = [(-score(root), 0, 0, [])]
    seen = {start_key}
    serial = 1
    limit = 1500
    best = (score(root), [])
    while queue and len(seen) < limit:
        _, depth, _, path = heapq.heappop(queue)
        if depth >= 10:
            continue
        node = reconstruct(root, path)
        for macro in choices(node):
            child_path = path + [macro]
            child = reconstruct(root, child_path)
            if child.levels_completed > 5:
                print("FOUND", child_path)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_score = score(child)
            if child_score > best[0]:
                best = (child_score, child_path)
                print("BEST", len(seen), child_score, child_path)
            heapq.heappush(queue, (-child_score, depth + 1, serial, child_path))
            serial += 1
    print("NOT_FOUND", len(seen), best)


A.run_program("m0r0", run)
