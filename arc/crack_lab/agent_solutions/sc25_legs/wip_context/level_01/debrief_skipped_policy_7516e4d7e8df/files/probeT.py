import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def reservoir(f): return int((f[:,62:64]==14).sum())
def show_ck(f):
    for r in range(47,64):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(22,39)))

def main(env):
    base=np.asarray(env.frame()).copy()
    print("base reservoir(cols62-63):",reservoir(base))
    empties=[(30,50),(25,55),(35,55),(30,60)]  # (col,row)
    # click a FILLED cell to see effect
    c=env.clone(); c.step(6,25,50); c.step(6,25,50)
    f=np.asarray(c.frame())
    print("click filled (25,50): res",reservoir(f),"lvl",c.levels_completed)
    print("  cell now:",''.join('.' if f[50,cc]==0 else format(f[50,cc],'x') for cc in range(24,27)))
    # fill all 4 empties one by one, settle at end
    c=env.clone()
    for (x,y) in empties:
        c.step(6,x,y)
    c.step(6,30,50) # settle
    f=np.asarray(c.frame())
    print("after filling all 4 empties: res",reservoir(f),"lvl",c.levels_completed,"term",c.terminal())
    show_ck(f)
A.run_program('sc25', main)
