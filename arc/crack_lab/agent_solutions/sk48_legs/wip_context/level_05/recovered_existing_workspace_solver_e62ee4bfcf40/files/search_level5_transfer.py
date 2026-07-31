"""Best-first macro search for a verified level-5 cross-block transfer."""
import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def pieces(env):
    return tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def heuristic(env, initial_eights):
    current = pieces(env)
    eights = tuple(piece for piece in current if piece[0] == 8)
    nines = tuple(piece for piece in current if piece[0] == 9)
    if eights != initial_eights:
        return 100000
    vertical_contacts = sum(
        1
        for _, nr, nc in nines
        for _, er, ec in eights
        if nc == ec and abs(nr - er) == 6
    )
    close = min(
        abs(nr - er) + abs(nc - ec)
        for _, nr, nc in nines
        for _, er, ec in eights
    )
    staged = len({row for _, row, _ in nines})
    reach = max(col for _, _, col in nines)
    return vertical_contacts * 100 + staged * 20 + reach - close


def search(root, max_states=6000, max_depth=14):
    initial_eights = tuple(piece for piece in pieces(root) if piece[0] == 8)
    queue = [(-heuristic(root, initial_eights), 0, 0, (), None, ())]
    seen = set()
    counter = 0
    best = (-1, (), ())
    while queue and len(seen) < max_states:
        _priority, depth, _, path, last_action, context = heapq.heappop(queue)
        node = p.replay(root, path)
        frame_key = p.arr(node.frame()).tobytes()
        key = (frame_key, context)
        if key in seen:
            continue
        seen.add(key)
        value = heuristic(node, initial_eights)
        if value > best[0]:
            best = (value, path, pieces(node))
            print("BEST", value, list(path), best[2], len(seen))
        if (
            node.levels_completed > root.levels_completed
            or tuple(piece for piece in pieces(node) if piece[0] == 8)
            != initial_eights
        ):
            return path, node, len(seen), best
        if depth >= max_depth:
            continue
        for action in (1, 2, 3, 4, 6):
            if action == last_action:
                continue
            branch = node.clone()
            for count in range(1, 7):
                branch.step(action)
                child_path = path + (action,) * count
                child_context = context
                if action in (1, 2):
                    child_context = ()
                elif action in (3, 4):
                    child_context = (context + (action,) * count)[-8:]
                counter += 1
                child_value = heuristic(branch, initial_eights)
                heapq.heappush(
                    queue,
                    (
                        -child_value + depth,
                        depth + 1,
                        counter,
                        child_path,
                        action,
                        child_context,
                    ),
                )
        if len(seen) % 500 == 0:
            print("PROGRESS", len(seen), depth, len(queue), best[0])
    return None, None, len(seen), best


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    path, node, seen, best = search(env)
    print("SEARCH", list(path) if path else None, seen, best)
    if node is not None:
        print("FOUND", node.levels_completed, pieces(node))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
