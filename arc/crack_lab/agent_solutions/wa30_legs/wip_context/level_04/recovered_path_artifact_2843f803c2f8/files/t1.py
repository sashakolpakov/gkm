from probe import *
env = get_env()
f0 = np.array(env.frame())
for a in [1,2,3,4,5]:
    c = env.clone()
    c.step(a)
    f1 = np.array(c.frame())
    ys, xs = np.where(f0 != f1)
    print(f"action {a}: {len(ys)} diffs", end='')
    if len(ys):
        print(f" bbox rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}")
        vals = set(zip(f0[ys,xs].tolist(), f1[ys,xs].tolist()))
        print("  changes:", sorted(vals))
    else:
        print()
