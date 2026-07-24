import sys
from collections import defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr, object_candidates


BUTTONS = ((20, 17), (14, 26), (14, 35), (39, 17), (48, 26), (48, 35))


def positions(env):
    return tuple((o["bbox"][0], o["bbox"][1])
                 for o in object_candidates(env.frame(), min_area=4)
                 if o["size"] == (2, 2) and o["area"] == 4 and o["bbox"][0] >= 16)


def state(env, spots):
    frame = arr(env.frame())
    return tuple(int(frame[r, c]) for r, c in spots)


def score(values, spots):
    cells = {spot: value for spot, value in zip(spots, values)}
    adjacent = sum(value == cells.get((r + dr, c + dc))
                   for (r, c), value in cells.items()
                   for dr, dc in ((0, 3), (3, 0)))
    groups = defaultdict(list)
    for spot, value in zip(spots, values):
        groups[value].append(spot)
    spread = sum(abs(r1 - r2) + abs(c1 - c2)
                 for same in groups.values()
                 for i, (r1, c1) in enumerate(same)
                 for r2, c2 in same[i + 1:])
    return adjacent * 100 - spread


def probe(env):
    for _ in range(5):
        env.step(6, 5, 32)
    base = env.clone()
    base_level = env.levels_completed
    spots = positions(env)
    inputs = []
    outputs = [[] for _ in BUTTONS]
    context = env.clone()
    walk = tuple((i * i + 3 * i + 1) % 6 for i in range(48))
    for advance in walk:
        inputs.append(state(context, spots))
        for i, (x, y) in enumerate(BUTTONS):
            child = context.clone()
            child.step(6, x, y)
            outputs[i].append(state(child, spots))
        x, y = BUTTONS[advance]
        context.step(6, x, y)
    signatures = defaultdict(list)
    for source in range(len(spots)):
        signatures[tuple(values[source] for values in inputs)].append(source)
    perms = []
    ambiguous = 0
    for button in range(len(BUTTONS)):
        perm = []
        for dest in range(len(spots)):
            signature = tuple(values[dest] for values in outputs[button])
            choices = signatures[signature]
            if len(choices) != 1:
                ambiguous += 1
            perm.append(choices[0])
        perms.append(tuple(perm))
    print("PERMS", len(spots), [len(set(p)) for p in perms], "ambiguous", ambiguous)

    initial = state(base, spots)
    beam = [(score(initial, spots), initial, ())]
    seen = {initial}
    candidates = []
    for depth in range(1, 31):
        nxt = []
        for _, values, path in beam:
            for button, perm in enumerate(perms):
                child = tuple(values[source] for source in perm)
                if child in seen:
                    continue
                seen.add(child)
                item = (score(child, spots), child, path + (button,))
                nxt.append(item)
                candidates.append(item)
        nxt.sort(key=lambda item: item[0], reverse=True)
        beam = nxt[:1500]
        if not beam:
            break
        print("DEPTH", depth, "best", beam[0][0], "path", beam[0][2],
              "seen", len(seen))

    candidates.sort(key=lambda item: item[0], reverse=True)
    checked_paths = set()
    for rank, (dense, _, path) in enumerate(candidates[:300]):
        if path in checked_paths:
            continue
        checked_paths.add(path)
        child = base.clone()
        for button in path:
            x, y = BUTTONS[button]
            child.step(6, x, y)
            if child.levels_completed > base_level:
                print("FOUND", path, "rank", rank, "dense", dense)
                return
    print("NO_GOAL", "checked", len(checked_paths), "best", candidates[0][0])


A.run_program("lp85", probe)
