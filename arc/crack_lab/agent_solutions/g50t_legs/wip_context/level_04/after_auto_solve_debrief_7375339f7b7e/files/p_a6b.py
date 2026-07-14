import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def diffcount(a,b): return int((a!=b).sum())
env=A.Arena('g50t')
# state B: toggle gates open
for a in [4,4,4,4,5]: env.step(a)
base=env.clone()
targets=[(46,52),(52,46),(41,10),(10,41),(2,2),(46,46),(31,31),(63,60),(0,0)]
print("From gates-open state, action6 diffs (>heartbeat):")
for (x,y) in targets:
    c=base.clone(); b=c.frame().copy(); c.step(6,x,y)
    d=diffcount(b,c.frame())
    if d>1 or c.levels_completed>0:
        print("click",(x,y),"diff",d,"lvl",c.levels_completed)
print("done B")
