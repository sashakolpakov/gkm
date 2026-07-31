import gkm_try as harness
from legs import move_vessel_below_and_apply, apply_current_then_select_and_apply_southeast
from perception import arr


TAIL = (3, 2, (53, 4), 5, 1, (29, 4), 5,
        4, 4, 2, (47, 4), 5, 1, 3, 3, (29, 4), 5)
OPS = (
    (3, 2, (53, 4), 5),
    (1, (29, 4), 5),
    (4, 4, 2, (47, 4), 5),
    (1, 3, 3, (29, 4), 5),
)


def step(node, action):
    if isinstance(action, tuple):
        node.step(6, action[0], action[1])
    else:
        node.step(action)


def pos_key(node):
    return (arr(node.frame())[18:63] == 2).tobytes()


def paths_from_north(env):
    q = [(env.clone(), ())]
    seen = {}
    while q:
        node, path = q.pop(0)
        key = pos_key(node)
        if key in seen or len(path) > 6:
            continue
        seen[key] = path
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + (action,)))
    return seen


def return_path(node, north_key):
    q = [(node.clone(), ())]
    seen = set()
    while q:
        cur, path = q.pop(0)
        key = pos_key(cur)
        if key == north_key:
            return path
        if key in seen or len(path) > 6:
            continue
        seen.add(key)
        for action in (1, 2, 3, 4):
            child = cur.clone()
            child.step(action)
            q.append((child, path + (action,)))


def probe(env):
    move_vessel_below_and_apply(env)
    apply_current_then_select_and_apply_southeast(env, 46, 4)
    target = arr(env.frame())[3:13, 3:13].copy()
    for insertion in range(5):
        prefix = sum(OPS[:insertion], ())
        suffix = sum(OPS[insertion:], ())
        at_prefix = env.clone()
        for action in prefix:
            step(at_prefix, action)
        prefix_key = pos_key(at_prefix)
        for _, outward in paths_from_north(at_prefix).items():
            node = at_prefix.clone()
            for action in outward:
                step(node, action)
            node.step(6, 35, 4)
            node.step(5)
            back = return_path(node, prefix_key)
            for action in back:
                step(node, action)
            for action in suffix:
                step(node, action)
            score = int((arr(node.frame())[34:44, 27:37] != target).sum())
            print("TRY", insertion, outward, "BACK", back, "SCORE", score,
                  "LEVEL", node.levels_completed)
            if node.levels_completed >= 3:
                plan = prefix + outward + ((35, 4), 5) + back + suffix
                print("FOUND", plan)
                return


if __name__ == "__main__":
    harness.A.run_program("cd82", probe)
