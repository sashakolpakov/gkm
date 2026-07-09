import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck=json.load(open('checkpoint.json'))
def fresh():
    e=A.Arena('wa30', _budget=A._Budget(200000))
    for a in ck['final_path']: e.step(a)
    return e
def classify(blk):
    s=set(blk.flatten().tolist()); s.discard(1)
    if not s: return '..'
    d={frozenset({5}):'##',frozenset({7}):'==',frozenset({2}):'22',
       frozenset({4}):'44',frozenset({12}):'CC',frozenset({15}):'MM',
       frozenset({9}):'99',frozenset({0}):'00',frozenset({0,14}):'AV',
       frozenset({14}):'ee',frozenset({4,9}):'4c',frozenset({2,9}):'2c',
       frozenset({1,2}):'22',frozenset({1,7}):'==',frozenset({5,7}):'#7'}
    return d.get(frozenset(s),'?'+''.join(str(x) for x in sorted(s)))
def show(f):
    for r in range(16):
        print(f'{r:2d} '+' '.join(classify(f[r*4:r*4+4,c*4:c*4+4]) for c in range(16)))
def c7(env): return int((np.asarray(env.frame())==7).sum())
LEFT=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
RIGHT=[(2,13),(2,14),(6,13),(6,14)]
SOCK=set(LEFT+RIGHT)
def cellset(env):
    f=np.asarray(env.frame()); box=set(); empty=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 9 in u and 2 in u and 4 not in u: empty.add((R,C))
            elif 9 in u and 4 in u and 2 not in u: box.add((R,C))
    return box,empty
def nfill(env,cl):
    box,_=cellset(env); return sum(1 for s in cl if s in box)
