import gkm_try as harness
from legs import move_vessel_below_and_apply, apply_current_then_select_and_apply_southeast
from perception import arr


def mask(a):
    return "/".join("".join("X" if int(v) else "." for v in row)
                    for row in a[34:44, 27:37])


def probe(env):
    move_vessel_below_and_apply(env)
    apply_current_then_select_and_apply_southeast(env, 46, 4)
    base = env.clone()
    q = [(base, ())]
    seen = set()
    found = {}
    while q:
        node, path = q.pop(0)
        a = arr(node.frame())
        key = a[18:63].tobytes()
        if key in seen or len(path) > 6:
            continue
        seen.add(key)
        used = node.clone()
        used.step(5)
        m = mask(arr(used.frame()))
        found.setdefault(m, path)
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + (action,)))
    print("STATES", len(seen), "STAMPS", len(found))
    for m, path in sorted(found.items(), key=lambda x: (len(x[1]), x[1])):
        print(path, m)


if __name__ == "__main__":
    harness.A.run_program("cd82", probe)
