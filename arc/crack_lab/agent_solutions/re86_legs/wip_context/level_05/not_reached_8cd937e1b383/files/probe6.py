import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH=json.load(f)["final_path"]

def solve(env):
    for a in PATH: env.step(a)
    f=arr(env.frame())
    bg=Counter(int(v) for v in f.flat).most_common(1)[0][0]
    scout=env.clone()
    first=None
    shapes=[]
    while True:
        bf=arr(scout.frame()).copy()
        bp=list(zip(*((bf==0).nonzero())))
        if len(bp)!=1: break
        ctr=tuple(int(v) for v in bp[0])
        if first is None: first=ctr
        elif ctr==first: break
        votes=[]; offs=set()
        for act in (1,2,3,4):
            m=scout.clone(); m.step(act); af=arr(m.frame())
            for r,c in zip(*((bf!=af).nonzero())):
                if int(af[r,c])==bg and int(bf[r,c]) not in (0,bg,2,4):
                    votes.append(int(bf[r,c]))
        col=Counter(votes).most_common(1)[0][0]
        for act in (1,2,3,4):
            m=scout.clone(); m.step(act); af=arr(m.frame())
            for r,c in zip(*((bf!=af).nonzero())):
                if int(bf[r,c])==col and int(af[r,c])==bg:
                    offs.add((int(r)-ctr[0],int(c)-ctr[1]))
        rr=[o[0] for o in offs]; cc=[o[1] for o in offs]
        axis=sum(a==0 or b==0 for a,b in offs)
        xs=sum(abs(a)==abs(b) for a,b in offs)
        dsum=[abs(a)+abs(b) for a,b in offs]
        print(f"shape ctr={ctr} color={col} npix={len(offs)} "
              f"r[{min(rr)},{max(rr)}] c[{min(cc)},{max(cc)}] "
              f"axisfrac={axis}/{len(offs)} xfrac={xs}/{len(offs)} maxdsum={max(dsum)}")
        # print classification hints
        print("   offsets sample:", sorted(offs)[:6], "...", sorted(offs)[-4:])
        shapes.append((ctr,col,offs))
        scout.step(5)
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
