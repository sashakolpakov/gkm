from probe5 import *
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
    else:
        print()
# watch courier over 8 idle steps (action 5 maybe not idle; use repeated up against wall?)
c = env.clone()
prev = np.array(c.frame())
for i in range(8):
    c.step(1)
    cur = np.array(c.frame())
    ys, xs = np.where(prev != cur)
    print(f"idle step {i}: diffs {len(ys)}", f"rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}" if len(ys) else "")
    prev = cur
