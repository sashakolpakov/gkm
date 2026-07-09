from pl9 import *
from legs import carry_box_to, _grid_scan, _movers
import perception as P
def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
def solve9(env, verbose=False):
    big8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
    east=[(5,7),(4,7),(3,7)]
    tset=set(big8)
    base=env.levels_completed
    def nbig():
        g=grid(np.asarray(env.frame())); return sum(1 for s in big8 if g[s[0]][s[1]] in('4c','?39'))
    while not env.terminal() and env.levels_completed<=base:
        if nbig()>=8: break
        av,boxes,walls=_grid_scan(env)
        ms=_movers(env,15)
        # clear mover if adjacent
        if ms:
            m=ms[0]
            if abs(av[0]-m[0])+abs(av[1]-m[1])==1:
                env.step(DIRS[(m[0]-av[0],m[1]-av[1])]); env.step(5)
                if verbose: print('cleared? mover now',_movers(env,15),'c7',c7(env))
                continue
        g=grid(np.asarray(env.frame()))
        open_east=[s for s in east if g[s[0]][s[1]] not in('4c','?39')]
        free_right=[b for b in boxes if b[1]>=10 and b not in tset]
        if open_east and free_right:
            best=None
            for b in free_right:
                for s in open_east:
                    d=abs(b[0]-s[0])+abs(b[1]-s[1])
                    if best is None or d<best[0]: best=(d,b,s)
            _,b,s=best
            if env.terminal(): break
            try: ok=carry_box_to(env,b,s)
            except Exception: break
            if verbose: print('east carry',b,'->',s,'ok',ok,'big',nbig(),'c7',c7(env))
            if not ok: env.step(5)
        else:
            env.step(5)  # yield for courier
    return nbig()
if __name__=='__main__':
    env=fresh()
    n=solve9(env, verbose=True)
    print('RESULT lvl',env.levels_completed,'big',n,'term',env.terminal())
