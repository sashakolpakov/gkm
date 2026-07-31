"""Focused probes from the two-contact level-5 frontier."""
import sys
import heapq

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


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    prefix = [3, 1] + [4] * 5
    root = p.replay(env, prefix)
    initial_eights = tuple(piece for piece in pieces(root) if piece[0] == 8)
    print("ROOT", pieces(root))
    runs = [()] + [
        (action,) * count
        for action in (1, 2, 3, 4, 6)
        for count in range(1, 7)
    ]
    best = []
    seen = set()
    for first in runs:
        for second in runs:
            path = first + second
            branch = p.replay(root, path)
            current_eights = tuple(
                piece for piece in pieces(branch) if piece[0] == 8
            )
            if (
                current_eights != initial_eights
                or branch.levels_completed > env.levels_completed
            ):
                print(
                    "FOUND",
                    list(path),
                    branch.levels_completed,
                    pieces(branch),
                )
                return
            state = pieces(branch)
            if state in seen:
                continue
            seen.add(state)
            nines = tuple(piece for piece in state if piece[0] == 9)
            contacts = sum(
                1
                for _, nr, nc in nines
                for _, er, ec in current_eights
                if nc == ec and abs(nr - er) == 6
            )
            rows = len({row for _, row, _ in nines})
            score = contacts * 100 + rows * 20 + max(
                col for _, _, col in nines
            )
            heapq.heappush(best, (score, list(path), state))
            if len(best) > 12:
                heapq.heappop(best)
    print("TWO_MACRO_NONE", len(seen))
    for item in sorted(best, reverse=True):
        print("BEST", item)


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
