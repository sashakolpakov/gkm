import l9env, perception as P
import numpy as np
env = l9env.get_l9()
base = np.asarray(env.frame())
frames={}
for a in [1,2,3,4,5]:
    c=env.clone(); c.step(a); frames[a]=np.asarray(c.frame())
# common change across all actions (self-movers/environment)
changed = {a:(frames[a]!=base) for a in frames}
common = np.ones_like(base,bool)
for a in frames: common &= changed[a]
print("common changed cells:", int(common.sum()))
ys,xs=np.where(common)
for y,x in list(zip(ys,xs))[:40]:
    print("  common",y,x,base[y,x],'->',frames[1][y,x])
# action-specific: cells that differ between actions
print("=== per action unique ===")
for a in frames:
    uniq = changed[a] & ~common
    ys,xs=np.where(uniq)
    print(P.ACTION_NAME[a],"uniq",int(uniq.sum()))
    for y,x in list(zip(ys,xs))[:20]:
        print("   ",y,x,base[y,x],'->',frames[a][y,x])
