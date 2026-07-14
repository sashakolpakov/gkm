import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def show(env,r,c):
    f=env.frame()
    for row in f[r[0]:r[1]]:
        print(''.join(str(int(v)) for v in row[c[0]:c[1]]))

env=A.Arena('g50t')
env.step(4)  # to (8,20)
print("BEFORE USE at (8,20):")
show(env,(6,26),(12,45))
env.step(5)
print("AFTER USE:")
show(env,(6,26),(12,45))
