from probe import *
env = get_env()
f0 = np.array(env.frame())
for a in [1,2,3,4]:
    c = env.clone()
    c.step(a)
    f1 = np.array(c.frame())
    ys, xs = np.where(f0 != f1)
    print(f"--- action {a}")
    for y, x in zip(ys, xs):
        print(f"  ({y},{x}): {f0[y,x]} -> {f1[y,x]}")
