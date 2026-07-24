import gkm_try as harness
from legs import move_vessel_below_and_apply, apply_current_then_select_and_apply_southeast
from perception import arr


CLICKS = ((23, 4), (29, 4), (35, 4), (41, 4), (47, 4), (53, 4), (59, 4))


def step(node, action):
    if isinstance(action, tuple):
        node.step(6, action[0], action[1])
    else:
        node.step(action)


def state_key(node):
    a = arr(node.frame())
    return a[:63].tobytes()


def placements(node):
    q = [(node.clone(), ())]
    seen = set()
    out = []
    while q:
        cur, path = q.pop(0)
        key = arr(cur.frame())[18:63].tobytes()
        if key in seen or len(path) > 6:
            continue
        seen.add(key)
        out.append((cur, path))
        for action in (1, 2, 3, 4):
            child = cur.clone()
            child.step(action)
            q.append((child, path + (action,)))
    return out


def probe(env):
    move_vessel_below_and_apply(env)
    apply_current_then_select_and_apply_southeast(env, 46, 4)
    base_level = env.levels_completed
    target = arr(env.frame())[3:13, 3:13].copy()
    beam = [(100, env.clone(), (), (0, 0, 0, 0))]
    for macro_depth in range(1, 8):
        candidates = {}
        for _, node, path, _ in beam:
            for placed, moves in placements(node):
                for click in CLICKS:
                    child = placed.clone()
                    child.step(6, click[0], click[1])
                    child.step(5)
                    child_path = path + moves + (click,) + (5,)
                    if child.levels_completed > base_level:
                        print("FOUND", child_path, "macros", macro_depth,
                              "actions", len(child_path))
                        return
                    work = arr(child.frame())[34:44, 27:37]
                    score = int((work != target).sum())
                    correct = tuple(int(((work == color) & (target == color)).sum())
                                    for color in (15, 12, 14, 8))
                    key = state_key(child)
                    old = candidates.get(key)
                    if old is None or (score, len(child_path)) < (old[0], len(old[2])):
                        candidates[key] = (score, child, child_path, correct)
        values = list(candidates.values())
        chosen = {}
        rankings = (
            lambda x: (x[0], len(x[2])),
            lambda x: (x[0] - 3 * x[3][1], len(x[2])),
            lambda x: (x[0] - 2 * x[3][2], len(x[2])),
            lambda x: (x[0] - 3 * x[3][3], len(x[2])),
        )
        for ranking in rankings:
            for item in sorted(values, key=ranking)[:20]:
                chosen[state_key(item[1])] = item
        beam = sorted(chosen.values(), key=lambda x: (x[0], len(x[2])))[:80]
        print("DEPTH", macro_depth, "BEST", [(x[0], x[2]) for x in beam[:3]],
              "UNIQUE", len(candidates))
    print("NOT FOUND")


if __name__ == "__main__":
    harness.A.run_program("cd82", probe)
