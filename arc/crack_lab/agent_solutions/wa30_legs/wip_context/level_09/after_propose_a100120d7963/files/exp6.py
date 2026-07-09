from l8env import l8
from legs import carry_box_to, _grid_scan, _bfs_path, _avatar_nav
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
botc=[(r,co) for r in (12,13,14) for co in (12,13,14)]
def empty_cont(f,cells):
    return [(R,C) for (R,C) in cells if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def free_bot(c):
    av,boxes,walls=_grid_scan(c)
    return av,[b for b in boxes if b[0]>9 and b not in botc]
c=l8().clone()
fail={}
for it in range(6):
    f=np.asarray(c.frame())
    av,fb=free_bot(c)
    fb=[b for b in fb if fail.get(b,0)<3]
    empt=empty_cont(f,botc)
    if not fb or not empt: break
    fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=fb[0]; empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
    ok=any(carry_box_to(c,box,t) for t in empt[:3])
    if not ok: fail[box]=fail.get(box,0)+1
    print('bot it%d box%s ok=%s mu=%d av=%s'%(it,box,ok,mu(c),_grid_scan(c)[0]))
av,boxes,walls=_grid_scan(c)
print('after bottom: avatar',av,'mu',mu(c))
blk=set(walls)|set(boxes); blk.discard(av)
p=_bfs_path(av,(3,3),blk)
print('bfs av->(3,3):',p)
print('boxes',sorted(boxes))
# is avatar boxed in? neighbors
for a,(dr,dc) in {1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}.items():
    nb=(av[0]+dr,av[1]+dc)
    print(' nb',nb,'wall' if nb in walls else ('box' if nb in boxes else 'free'))
