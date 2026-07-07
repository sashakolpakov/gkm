import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def make6():
    ck = json.load(open("checkpoint.json"))
    env = A.Arena('ls20')
    for a in ck["final_path"]:
        env.step(a)
    return env

def st(f):
    f = np.array(f)
    rs, cs = np.where(f == 12)
    av = None
    for r, c in zip(rs, cs):
        if c+4 < 64 and (f[r, c:c+5] == 12).all():
            av = (int(r), int(c)); break
    # portrait color: any of 14,8,12,9,0 in rows 55-60 cols 2-9
    reg = f[55:61, 2:10]
    pcol = [int(v) for v in np.unique(reg) if v != 5]
    bar = int((f[60:63] == 11).sum())
    pips4 = int((f[4] == 1).sum())
    door = int((f[20:25, 54] == 1).sum())
    tr = [int(v) for v in np.unique(f[34:41, 53:60]) if v != 5]   # top-right room
    br = [int(v) for v in np.unique(f[48:56, 53:60]) if v != 5]   # bottom-right room
    # prey positions
    rs, cs = np.where(f == 14)
    cr = [(int(r),int(c)) for r,c in zip(rs,cs) if r < 55 and 5 <= c <= 50]
    rs, cs = np.where((f==0)|(f==1))
    sa = [(int(r),int(c)) for r,c in zip(rs,cs) if 10<=r<=14 and 9<=c<=47]
    sb = [(int(r),int(c)) for r,c in zip(rs,cs) if 40<=r<=44 and 9<=c<=47]
    boxes = int((f[5:50, 9:47] == 11).sum())
    return dict(av=av, pcol=pcol, bar=bar, pips4=pips4, door=door, tr=tr, br=br,
                cr=(min(cr) if cr else None), sa=(min(sa) if sa else None),
                sb=(min(sb) if sb else None), boxes=boxes)
