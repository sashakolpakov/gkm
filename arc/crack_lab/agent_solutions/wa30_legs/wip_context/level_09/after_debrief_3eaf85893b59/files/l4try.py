import sys, json, random
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import legs as L
_p=json.load(open('base_shrunk.json'))
_base=None
def fresh():
    global _base
    if _base is None:
        e=A.Arena('wa30', _budget=A._Budget(600*400))
        for a in _p[:160]: e.step(a)
        _base=e
    return _base.clone()
UP,DOWN,LEFT,RIGHT=1,2,3,4
specs=[
    ([(RIGHT,1)],DOWN,DOWN,2),
    ([(UP,1),(RIGHT,1)],DOWN,DOWN,1),
    ([(LEFT,2)],LEFT,LEFT,1),
    ([(UP,3),(RIGHT,1),(UP,1)],LEFT,LEFT,1),
    ([(DOWN,1),(RIGHT,3)],UP,DOWN,2),
    ([(LEFT,1),(UP,1)],RIGHT,RIGHT,2),
    ([(UP,1),(LEFT,3),(UP,1)],RIGHT,RIGHT,3),
]
def run(order):
    e=fresh()
    try:
        L.ferry_all_then_yield(e,[specs[i] for i in order],DOWN)
    except Exception:
        return None
    if e.levels_completed>=4:
        return len(e.path)-160
    return None
base=run(list(range(7)))
print('base order len',base)
best=(base,list(range(7)))
random.seed(1)
import itertools, time
t0=time.time()
tried=set()
while time.time()-t0<240:
    o=list(range(7)); random.shuffle(o)
    k=tuple(o)
    if k in tried: continue
    tried.add(k)
    r=run(o)
    if r is not None and r<best[0]:
        best=(r,o); print('better',r,o,flush=True)
print('BEST',best)
