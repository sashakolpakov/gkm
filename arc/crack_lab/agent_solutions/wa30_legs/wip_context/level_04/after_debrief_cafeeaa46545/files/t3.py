from probe import *
env = get_env()
c = env.clone()
# walk down repeatedly to see movement and timer expiry
for i in range(16):
    c.step(2)
    f = np.array(c.frame())
    n7 = (f[63]==7).sum()
    # avatar position
    ys, xs = np.where(f==14)
    pos = (ys.min(), xs.min()) if len(ys) else None
    print(i, "timer7s:", n7, "avatar14 bbox:", pos, "levels:", c.levels_completed, "term:", c.terminal())
