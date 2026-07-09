from pl9 import *
from legs import carry_box_to, grab_and_deliver, _grid_scan, _movers, _bfs_path
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
BIG9=[(4,6),(3,7),(4,7),(5,7),(3,5),(5,5),(3,6),(5,6),(4,5)]
def solve9(env, verbose=False):
    tset=set(BIG9)
    base=env.levels_completed
    mover_gone=False
    def g(): return grid(np.asarray(env.frame()))
    def filled(gr,s): return gr[s[0]][s[1]] in('4c','?39')
    def nfill(): 
        gr=g(); return sum(1 for s in BIG9 if filled(gr,s))
    guard=0
    while not env.terminal() and env.levels_completed<=base and guard<200:
        guard+=1
        if nfill()>=9: break
        av,boxes,walls=_grid_scan(env)
        ms=_movers(env,15)
        if ms:
            m=ms[0]; d=abs(av[0]-m[0])+abs(av[1]-m[1])
            if d==1:
                env.step(DIRS[(m[0]-av[0],m[1]-av[1])]); env.step(5)
                if not _movers(env,15): mover_gone=True
                if verbose: print('USE mover; gone?',mover_gone,'c7',c7(env)); 
                continue
            if m[0]<=9:  # mover loose in reachable region -> chase to clear
                blocked=(set(walls)|set(boxes))-{m}
                p=_bfs_path(av,m,blocked)
                if p: env.step(p[0]); continue
                # can't path (maze); step toward it greedily
                best=None
                for (dr,dc),a in DIRS.items():
                    nb=(av[0]+dr,av[1]+dc)
                    if not(0<=nb[0]<16 and 0<=nb[1]<16) or nb in walls or nb in boxes: continue
                    dd=abs(nb[0]-m[0])+abs(nb[1]-m[1])
                    if best is None or dd<best[0]: best=(dd,a)
                if best: env.step(best[1]); continue
        # mover safe (in maze) or gone: fill a seat
        gr=g()
        opens=[s for s in BIG9 if not filled(gr,s)]
        free=[b for b in boxes if b not in tset]
        if not opens or not free: env.step(5); continue
        opens.sort(key=lambda s:(0 if s==(4,6) else 1, min(abs(b[0]-s[0])+abs(b[1]-s[1]) for b in free)))
        progressed=False
        for s in opens:
            if env.terminal(): break
            cand=sorted(free,key=lambda b:abs(b[0]-s[0])+abs(b[1]-s[1]))
            for b in cand[:5]:
                if env.terminal(): break
                # abort carry method if mover would come; just attempt
                try: ok=carry_box_to(env,b,s)
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
