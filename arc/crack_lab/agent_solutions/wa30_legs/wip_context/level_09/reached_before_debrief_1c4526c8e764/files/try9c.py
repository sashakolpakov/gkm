from pl9 import *
from legs import carry_box_to, _grid_scan, _movers, _bfs_path, _avatar_nav
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
BIG9=[(4,6),(3,7),(5,7),(4,7),(5,5),(3,5),(3,6),(5,6),(4,5)]
def solve9(env, verbose=False):
    tset=set(BIG9)
    base=env.levels_completed
    def g(): return grid(np.asarray(env.frame()))
    def filled(gr,s): return gr[s[0]][s[1]] in('4c','?39')
    def nfill(): 
        gr=g(); return sum(1 for s in BIG9 if filled(gr,s))
    def mover_in_left(av,m):
        # reachable region: rows<=9 and cols<=9 (left+bottomright open area)
        return m[0]<=9
    fail=set()
    while not env.terminal() and env.levels_completed<=base:
        if nfill()>=9: break
        ms=_movers(env,15); av,boxes,walls=_grid_scan(env)
        if ms:
            m=ms[0]
            d=abs(av[0]-m[0])+abs(av[1]-m[1])
            if d==1:
                env.step(DIRS[(m[0]-av[0],m[1]-av[1])]); env.step(5)
                if verbose: print('CLEARED mover ->',_movers(env,15),'c7',c7(env)); 
                fail=set(); continue
            if d<=4 and mover_in_left(av,m):
                # chase 1 step toward mover
                blocked=(set(walls)|set(boxes))-{m}
                p=_bfs_path(av,m,blocked)
                if p: env.step(p[0]); continue
        gr=g()
        target=None
        for s in BIG9:
            if not filled(gr,s) and s not in fail: target=s; break
        if target is None:
            fail=set(); 
            # all remaining are 'fail'; just yield to let courier
            env.step(5); continue
        free=[b for b in boxes if b not in tset]
        if not free: env.step(5); continue
        free.sort(key=lambda b:abs(b[0]-target[0])+abs(b[1]-target[1]))
        if env.terminal(): break
        done=False
        for b in free:
            try: done=carry_box_to(env,b,target)
            except Exception: done=False; break
            if done: break
        if verbose: print('fill',target,'done',done,'nfill',nfill(),'c7',c7(env),'mover',_movers(env,15))
        if not done:
            fail.add(target); env.step(5)
        else:
            fail=set()
    return nfill()
if __name__=='__main__':
    import sys
    env=fresh()
    n=solve9(env, verbose=True)
    print('RESULT lvl',env.levels_completed,'nfill',n,'term',env.terminal())
