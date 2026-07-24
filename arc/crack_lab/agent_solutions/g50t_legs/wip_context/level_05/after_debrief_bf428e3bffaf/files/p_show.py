import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def show(env,r=(0,64),c=(0,64)):
    f=env.frame()
    for row in f[r[0]:r[1]]:
        print(''.join(str(int(v)) for v in row[c[0]:c[1]]))

env=A.Arena('g50t')
for act in [4,4,4,4]:
    env.step(act)
print("after RIGHT x4 (avatar at 8,38 region):")
show(env,(6,16),(12,50))
print("USE now:")
b=env.frame().copy()
env.step(5)
print("changed pix:", (env.frame()!=b).sum())
