from pl9 import *
from legs import carry_box_to, _grid_scan, _movers, chase_and_clear
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
BIG9=[(4,6),(3,5),(3,6),(4,5),(5,5),(5,6),(3,7),(4,7),(5,7)]
def solve9(env, verbose=False):
    tset=set(BIG9); base=env.levels_completed; cleared=False
    def g(): return grid(np.asarray(env.frame()))
    def filled(gr,s): return gr[s[0]][s[1]] in('4c','?39')
    def nfill(): gr=g(); return sum(1 for s in BIG9 if filled(gr,s))
    guard=0
    while not env.terminal() and env.levels_completed<=base and guard<60:
        guard+=1
        if nfill()>=9: break
        ms=_movers(env,15)
        if ms and ms[0][0]<=9 and not cleared:
            # mover loose in left region -> clear it now
            chase_and_clear(env,15,lambda m:True,cap=12)
            if not _movers(env,15): cleared=True
            if verbose: print('chase-clear; c15gone?',not _movers(env,15),'c7',c7(env),'nfill',nfill())
            continue
        gr=g(); av,boxes,walls=_grid_scan(env)
        opens=[s for s in BIG9 if not filled(gr,s)]
        free=[b for b in boxes if b not in tset]
        if not opens or not free: env.step(5); continue
        # pick the open seat+box with min carry distance
        opens.sort(key=lambda s:(0 if s==(4,6) else 1, min(abs(b[0]-s[0])+abs(b[1]-s[1]) for b in free)))
        prog=False
        for s in opens:
            if env.terminal(): break
            for b in sorted(free,key=lambda b:abs(b[0]-s[0])+abs(b[1]-s[1]))[:5]:
                if env.terminal(): break
                try: ok=carry_box_to(env,b,s)
                except Exception: ok=False; break
                if ok: prog=True; break
            if prog:
                if verbose: print('filled',s,'nfill',nfill(),'c7',c7(env),'mover',_movers(env,15))
                break
        if not prog: env.step(5)
    return nfill()
if __name__=='__main__':
    env=fresh(); n=solve9(env,verbose=True)
    print('RESULT lvl',env.levels_completed,'nfill',n,'term',env.terminal())
