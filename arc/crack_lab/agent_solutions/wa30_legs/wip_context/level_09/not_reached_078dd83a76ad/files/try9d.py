from pl9 import *
from legs import carry_box_to, grab_and_deliver, _grid_scan, _movers, _cells
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
BIG9=[(4,6),(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
def solve9(env, verbose=False):
    tset=set(BIG9)
    base=env.levels_completed
    def g(): return grid(np.asarray(env.frame()))
    def filled(gr,s): return gr[s[0]][s[1]] in('4c','?39')
    def nfill():
        gr=g(); return sum(1 for s in BIG9 if filled(gr,s))
    def clear_if_adj():
        ms=_movers(env,15)
        if not ms: return False
        av=_grid_scan(env)[0]; m=ms[0]
        if abs(av[0]-m[0])+abs(av[1]-m[1])==1:
            env.step(DIRS[(m[0]-av[0],m[1]-av[1])]); env.step(5); return True
        return False
    rounds=0
    while not env.terminal() and env.levels_completed<=base and rounds<40:
        rounds+=1
        if nfill()>=9: break
        if clear_if_adj():
            if verbose: print('cleared',_movers(env,15),'c7',c7(env)); continue
        gr=g(); av,boxes,walls=_grid_scan(env)
        opens=[s for s in BIG9 if not filled(gr,s)]
        free=[b for b in boxes if b not in tset]
        if not opens or not free: env.step(5); continue
        # match: for each open seat try nearest boxes; commit first success
        progressed=False
        # order seats: center first, then by nearest free box distance
        opens.sort(key=lambda s:(0 if s==(4,6) else 1, min(abs(b[0]-s[0])+abs(b[1]-s[1]) for b in free)))
        for s in opens:
            if env.terminal(): break
            cand=sorted(free,key=lambda b:abs(b[0]-s[0])+abs(b[1]-s[1]))
            for b in cand[:4]:
                if env.terminal(): break
                try: ok=carry_box_to(env,b,s)
                except Exception: ok=False; break
                if ok: progressed=True; break
            if not progressed:
                for b in cand[:3]:
                    if env.terminal(): break
                    try: ok=grab_and_deliver(env,b,s)
                    except Exception: ok=False; break
                    if ok: progressed=True; break
            if progressed:
                if verbose: print('filled',s,'nfill',nfill(),'c7',c7(env),'mover',_movers(env,15))
                break
        if not progressed:
            env.step(5)
    return nfill()
if __name__=='__main__':
    env=fresh()
    n=solve9(env, verbose=True)
    print('RESULT lvl',env.levels_completed,'nfill',n,'term',env.terminal())
