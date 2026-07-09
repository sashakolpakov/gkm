import l9env, legs
import numpy as np
# Drive center i2 down using avatar grabbing center content? Let's watch what actions reduce i2 fastest greedily.
env=l9env.get_l9()
base=env.levels_completed
def i2(e):
    f=np.asarray(e.frame()); return int((f[13:23,21:31]==2).sum())
print("start i2",i2(env))
for step in range(60):
    if env.terminal() or env.levels_completed>base: break
    best=None
    for a in (1,2,3,4,5):
        c=env.clone(); c.step(a)
        if c.levels_completed>base:
            best=(a,-999); print("WIN via",a); break
        if c.terminal(): continue
        v=i2(c)
        if best is None or v<best[1]:
            best=(a,v)
    env.step(best[0])
    if step%5==0:
        print(f"step{step} act{best[0]} i2{i2(env)} loose{len(legs._grid_scan(env)[1])} n7{int((np.asarray(env.frame())==7).sum())}")
print("FINAL i2",i2(env),"lvl",env.levels_completed,"term",env.terminal())
# dump center
f=np.asarray(env.frame())
for r in range(12,24):
    print(''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(20,32)))
