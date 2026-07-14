import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
env=A.Arena('g50t')
b=env.frame()
hits=[]
for x in range(0,64,2):
    for y in range(0,64,2):
        c=env.clone(); c.step(6,x,y)
        d=(c.frame()!=b).sum()
        if d>0 or c.levels_completed>0:
            hits.append((x,y,int(d),c.levels_completed))
print("num hits",len(hits))
for h in hits[:60]: print(h)
