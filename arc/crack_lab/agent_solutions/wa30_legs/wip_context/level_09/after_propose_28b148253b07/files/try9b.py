from pl9 import *
from legs import carry_box_to, _grid_scan, _movers
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
BIG9=[(4,6),(3,7),(5,7),(4,7),(3,6),(5,6),(3,5),(5,5),(4,5)]
def solve9(env, verbose=False):
    tset=set(BIG9)
    base=env.levels_completed
    def filled(s):
        g=grid(np.asarray(env.frame())); return g[s[0]][s[1]] in('4c','?39')
    def nfill(): return sum(1 for s in BIG9 if filled(s))
    def try_clear():
        ms=_movers(env,15)
        if not ms: return False
        av=_grid_scan(env)[0]; m=ms[0]
        if abs(av[0]-m[0])+abs(av[1]-m[1])==1:
            env.step(DIRS[(m[0]-av[0],m[1]-av[1])]); env.step(5); return True
        return False
    while not env.terminal() and env.levels_completed<=base:
        if nfill()>=9: break
        if try_clear(): 
            if verbose: print('cleared mover',_movers(env,15),'c7',c7(env)); continue
        av,boxes,walls=_grid_scan(env)
        # next open seat in priority order
        target=None
        for s in BIG9:
            if not filled(s): target=s; break
        if target is None: break
        free=[b for b in boxes if b not in tset]
        if not free: env.step(5); continue
        free.sort(key=lambda b:abs(b[0]-target[0])+abs(b[1]-target[1]))
        if env.terminal(): break
        done=False
        for b in free[:2]:
            try: done=carry_box_to(env,b,target)
            except Exception: done=False; break
            if done: break
        if verbose: print('fill',target,'done',done,'nfill',nfill(),'c7',c7(env),'mover',_movers(env,15))
        if not done: env.step(5)
    return nfill()
if __name__=='__main__':
    env=fresh()
    n=solve9(env, verbose=True)
    print('RESULT lvl',env.levels_completed,'nfill',n,'term',env.terminal())
